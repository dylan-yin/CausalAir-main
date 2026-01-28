import os
import argparse
import torch
from tqdm import tqdm
import collections
import data_loader as module_data
import criterion as module_loss
import evaluation.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from utils import inf_loop, MetricTracker
from utils.dist import synchronize, get_rank, all_gather


def main(args, config):
    logger = config.get_logger('test')

    # prepare for (multi-device) GPU testing
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        synchronize()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, args=args)

    device = torch.device("cuda")
    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if update BatchNorm stats
            # broadcast_buffers=False,
        )

    # save file flag
    save_to_disk = get_rank() == 0

    # setup data_loader instances
    data_loader = config.init_obj('train_loader', module_data, args=args)
    valid_data_loader = config.init_obj('valid_loader', module_data, args=args)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if save_to_disk:
        logger.info(model)
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    if distributed:
        state_dict = {'module.' + k:v for k,v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict,strict=False)

    # prepare model for testing
    model.eval()
    total_loss = 0.0
    outputs = []
    targets = []
    # 处理valid_data_loader中的metedata和AQdata的缩放（测试时需要和训练时一致）
    if hasattr(valid_data_loader.dataset, 'metedata'):
        if data_loader.dataset.scale:
            L, N, C = valid_data_loader.dataset.metedata.shape
            metedata = valid_data_loader.dataset.metedata.reshape(L * N, C)
            metedata = data_loader.dataset.mete_scaler.transform(metedata)
            valid_data_loader.dataset.metedata = metedata.reshape(L, N, C)
    if hasattr(valid_data_loader.dataset, 'AQdata'):
        if data_loader.dataset.scale:
            L, N, C = valid_data_loader.dataset.AQdata.shape
            AQdata = valid_data_loader.dataset.AQdata.reshape(L * N, C)
            AQdata[:, -7:] = data_loader.dataset.aq_scaler.transform(AQdata[:, -7:])
            valid_data_loader.dataset.AQdata = AQdata.reshape(L, N, C)

    val_mean = torch.Tensor(data_loader.dataset.aq_scaler.mean_).to(device)
    val_var = torch.sqrt(torch.Tensor(data_loader.dataset.aq_scaler.var_).to(device))   
    valid_metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns])


    with torch.no_grad():
        pbar = tqdm(enumerate(valid_data_loader),
            total=int(len(valid_data_loader.sampler) / valid_data_loader.batch_size) + 1, leave=False)
        for batch_idx, (data, target) in pbar:

            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            for key, value in target.items():
                if torch.is_tensor(value):
                    target[key] = value.to(device)
        # target = target.to(self.device)

            model_outputs = model(data)
            if len(model_outputs) == 5:
                # STformer_v9_9 with dynamic causal matrices
                output, _, _, total_sparsity_loss, dynamic_causal_matrices = model_outputs
                
                # Calculate and update static sparsity (from theta matrix)
                if hasattr(model, 'theta'):
                    prob_matrix = torch.sigmoid(model.theta)
                    static_sparsity = torch.sum(torch.abs(prob_matrix)).item()
                    valid_metrics.update('Static_L1_Sparsity', static_sparsity)
                
                # Calculate and update dynamic sparsity
                if isinstance(dynamic_causal_matrices, list):
                    # For models that return a list of matrices (like v9_9)
                    dynamic_sparsity = sum(torch.sum(torch.abs(dcm)).item() for dcm in dynamic_causal_matrices)
                else:
                    # For models that return a single matrix (like v9_11, v9_18)
                    dynamic_sparsity = torch.sum(torch.abs(dynamic_causal_matrices)).item()
                valid_metrics.update('Dynamic_L1_Sparsity', dynamic_sparsity)
                
            elif len(model_outputs) == 4:
                # STformer_v9_6/v9_8 with sparsity loss
                output, _, _, sparsity_loss = model_outputs
                
                # Calculate and update static sparsity (from theta matrix)
                if hasattr(model, 'theta'):
                    prob_matrix = torch.sigmoid(model.theta)
                    static_sparsity = torch.sum(torch.abs(prob_matrix)).item()
                    valid_metrics.update('Static_L1_Sparsity', static_sparsity)
                
            elif len(model_outputs) == 3:
                output, _, _ = model_outputs
            elif len(model_outputs) == 2:
                # STformer_v9_7 with only prediction and initial prediction
                output, _ = model_outputs
            else:
                output, _ = model_outputs

            output = output*val_var + val_mean
            target =  target['label']*val_var + val_mean
            # output = self.data_loader.dataset.aq_scaler.inverse_transform(output)
            # target = self.data_loader.dataset.aq_scaler.inverse_transform(target['label'])
            for met in metric_fns:
                # Skip sparsity metrics as they are calculated separately
                if met.__name__ not in ['Static_L1_Sparsity', 'Dynamic_L1_Sparsity']:
                    valid_metrics.update(met.__name__, met(output, target))
            # outputs.append(output.clone()[:, :])
            # targets.append(target.clone())
            # attns.append(attn[0])
            # if batch_idx == self.len_epoch:
            #     break
    log = {}
    log.update({
        met.__name__: valid_metrics[i].item() for i, met in enumerate(metric_fns)
    })
    if save_to_disk:
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-l', '--local_rank', default=0, type=int,
                      help='local rank of gpu')
    args.add_argument('-s', '--save_dir', default=None, type=str,
                      help='dir of save path')
    args.add_argument('--seed', type=int, default=None, help='Random seed.')
    args.add_argument('--deterministic', action='store_true',
        help='Whether to set deterministic options for CUDNN backend.')
    args.add_argument('--debug', action='store_true',
                      help='Whether not to evaluate the checkpoint during training.')
    args.add_argument('--seq_len', type=int, default=72, help='input sequence length')
    args.add_argument('--label_len', type=int, default=72, help='start token length')
    args.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='train_data_loader;args;batch_size')
    ]
    args, config = ConfigParser.from_args(args, options, training=True)
    args.seq_len = config["arch"]["args"]["configs"]["seq_len"]
    args.pred_len = config["arch"]["args"]["configs"]["pred_len"]
    # fix random seeds for reproducibility
    if not args.seed is None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    assert not config.resume is None, "Testing mode requires model path!"
    main(args, config)
