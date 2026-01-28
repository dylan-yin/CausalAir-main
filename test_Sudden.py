import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import collections

import data_loader as module_data
import criterion as module_loss
import evaluation.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from utils import MetricTracker
from utils.dist import synchronize, get_rank


def main(args, config):
    logger = config.get_logger('test_sudden')

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    # save file flag
    save_to_disk = get_rank() == 0

    # setup data_loader instances
    # train_loader is used to obtain scalers fit on training data
    train_loader = config.init_obj('train_loader', module_data, args=args)

    # allow overriding the validation data path via CLI by mutating config before init
    if args.val_data is not None:
        try:
            config['valid_loader']['args']['data_dir'] = args.val_data
        except Exception:
            pass
    valid_loader = config.init_obj('valid_loader', module_data, args=args)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if save_to_disk:
        logger.info(model)
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location='cpu')
    if distributed:
        state_dict = {'module.' + k: v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint['state_dict']
    # remove potentially problematic positional embedding key if present
    if 'enc_embedding.spatial_embedding.pe' in state_dict:
        del state_dict['enc_embedding.spatial_embedding.pe']
    model.load_state_dict(state_dict, strict=False)

    # prepare model for testing
    model.eval()

    # Ensure valid set uses training scalers for consistency
    if hasattr(valid_loader.dataset, 'metedata'):
        if train_loader.dataset.scale:
            L, N, C = valid_loader.dataset.metedata.shape
            metedata = valid_loader.dataset.metedata.reshape(L * N, C)
            metedata = train_loader.dataset.mete_scaler.transform(metedata)
            valid_loader.dataset.metedata = metedata.reshape(L, N, C)
    if hasattr(valid_loader.dataset, 'AQdata'):
        if train_loader.dataset.scale:
            L, N, C = valid_loader.dataset.AQdata.shape
            AQdata = valid_loader.dataset.AQdata.reshape(L * N, C)
            AQdata[:, -7:] = train_loader.dataset.aq_scaler.transform(AQdata[:, -7:])
            valid_loader.dataset.AQdata = AQdata.reshape(L, N, C)

    # inverse transform params (only for last 7 AQ variables)
    val_mean = torch.tensor(train_loader.dataset.aq_scaler.mean_, dtype=torch.float32, device=device)
    val_std = torch.sqrt(torch.tensor(train_loader.dataset.aq_scaler.var_, dtype=torch.float32, device=device))

    # trackers
    valid_metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns])

    # extra PM2.5 metrics tracked separately
    extra_metrics = MetricTracker('PM25_MAE', 'PM25_RMSE')

    total_selected = 0
    total_pairs = 0
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(valid_loader),
            total=int(len(valid_loader.sampler) / valid_loader.batch_size) + 1,
            leave=False
        )
        for batch_idx, (data, target) in pbar:
            # move tensors to device
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            for key, value in target.items():
                if torch.is_tensor(value):
                    target[key] = value.to(device)

            # forward pass (support multiple return variants)
            model_outputs = model(data)
            if isinstance(model_outputs, (list, tuple)):
                output = model_outputs[0]
            else:
                output = model_outputs

            # inverse transform prediction and label (AQ last 7 variables)
            output = output * val_std + val_mean
            target_pred = target['label'] * val_std + val_mean

            # construct PM2.5 filter across seq_len + pred_len using input AQ series
            # data['aq_train_data']: [B, seq_len+pred_len, N, C]
            aq_series = data['aq_train_data']
            B, T, N, C = aq_series.shape
            # PM2.5 is the 2nd of last 7 AQ vars: index -6
            pm25_scaled = aq_series[..., -6]
            pm25_mean = val_mean[1].view(1, 1, 1)
            pm25_std = val_std[1].view(1, 1, 1)
            pm25_real = pm25_scaled * pm25_std + pm25_mean  # [B, T, N]

            max_pm25 = pm25_real.max(dim=1).values  # [B, N]
            min_pm25 = pm25_real.min(dim=1).values  # [B, N]
            amp_pm25 = max_pm25 - min_pm25          # [B, N]
            mask = (max_pm25 > 50.0) & (amp_pm25 > 20.0)  # [B, N]

            total_selected += mask.sum().item()
            total_pairs += mask.numel()

            # flatten mask to match output/target which are [B*N, pred_len, 7]
            mask_vec = mask.reshape(-1)
            if mask_vec.any():
                idx = torch.nonzero(mask_vec, as_tuple=False).squeeze(-1)
                filtered_output = output[idx]
                filtered_target = target_pred[idx]

                # update configured metrics (skip sparsity names if present in config)
                for met in metric_fns:
                    name = met.__name__
                    if name not in ['Static_L1_Sparsity', 'Dynamic_L1_Sparsity']:
                        valid_metrics.update(name, met(filtered_output, filtered_target))

                # extra PM2.5 metrics on filtered subset
                pm25_out = filtered_output[:, :, 1]
                pm25_tgt = filtered_target[:, :, 1]
                extra_metrics.update('PM25_MAE', F.l1_loss(pm25_out, pm25_tgt))
                extra_metrics.update('PM25_RMSE', torch.sqrt(F.mse_loss(pm25_out, pm25_tgt)))

                # loss over filtered subset
                loss = loss_fn(filtered_output, filtered_target)
                total_loss += loss.item() * pm25_out.shape[0]
            else:
                # no selected samples in this batch, skip metric and loss updates
                continue

    # prepare logs
    selection_ratio = (total_selected / total_pairs) if total_pairs > 0 else 0.0
    log = {
        'selected_pairs': total_selected,
        'total_pairs': total_pairs,
        'selection_ratio': selection_ratio,
    }

    # merge metric results
    val_log = valid_metrics.result()
    log.update(**{'val_' + k: v for k, v in val_log.items()})
    extra_log = extra_metrics.result()
    log.update(**extra_log)

    # average loss per selected pair (if any)
    if total_selected > 0:
        log['loss'] = total_loss / total_selected
    else:
        log['loss'] = None

    if save_to_disk:
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Sudden PM2.5 Event Testing')
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
    args.add_argument('--val_data', type=str, default='/mnt/hyin/Datasets/china_stations_data/aq_level_mete/val_data.pkl',
                      help='Path to validation data pkl for sudden test')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='train_loader;args;batch_size')
    ]
    args, config = ConfigParser.from_args(args, options, training=False)

    # sync seq/pred from config arch
    args.seq_len = config["arch"]["args"]["configs"]["seq_len"]
    args.pred_len = config["arch"]["args"]["configs"]["pred_len"]

    # fix random seeds for reproducibility
    import numpy as np
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    assert config.resume is not None, "Testing mode requires model path!"
    main(args, config)