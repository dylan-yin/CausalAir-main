import os
import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
import criterion as module_loss
import evaluation.metric as module_metric
import model as module_arch
import config as module_config
from parse_config import ConfigParser
from trainer import Trainer

from utils.dist import synchronize, get_rank


def main(args, config):
    logger = config.get_logger('train')

    # prepare for (multi-device) GPU training
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
    # if config['arch'].get('config', False):
    #     module_args = dict(config['arch']['config']['args'])
    #     #    'Overwriting kwargs given in config file is not allowed'
    #     modelconfig = getattr(module_config, config['arch']['config']['type'])(**module_args)
    #     model = config.init_obj('arch', module_arch, configs=modelconfig)
    # else:
    model = config.init_obj('arch', module_arch, args=args)


    device = torch.device("cuda")
    model.to(device)

    # get function handles of loss and metrics
    # criterion = config.init_obj('loss', module_loss)
    criterion = getattr(module_loss, config['loss'])()

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

   

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
    if not args.no_validate:
        valid_data_loader = config.init_obj('valid_loader', module_data, args=args)
    else:
        valid_data_loader = None

    if save_to_disk:
        logger.info(model)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      args=args,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-l', '--local_rank', default=0, type=int,
                      help='local rank of gpu')
    args.add_argument('-s', '--save_dir', default=None, type=str,
                      help='dir of save path')
    args.add_argument('--no-validate', action='store_true',
        help='Whether not to evaluate the checkpoint during training.')

    args.add_argument('--debug', action='store_true',
                      help='Whether not to evaluate the checkpoint during training.')


    args.add_argument('--seed', type=int, default=None, help='Random seed.')
    args.add_argument('--deterministic', action='store_true',
        help='Whether to set deterministic options for CUDNN backend.')

    # parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # # basic config
    # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    # parser.add_argument('--model', type=str, required=True, default='Autoformer',
    #                     help='model name, options: [Autoformer, Informer, Transformer]')
    #
    # # data loader
    # parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    # parser.add_argument('--features', type=str, default='M',
    #                     help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    # parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    args.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    args.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    args.add_argument('--label_len', type=int, default=24, help='start token length')
    args.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

    # model define
    args.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    args.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    args.add_argument('--c_out', type=int, default=7, help='output size')
    args.add_argument('--d_model', type=int, default=64, help='dimension of model')

    args.add_argument('--gat_node_features', type=int, default=7, help='num of heads')
    args.add_argument('--gat_hidden_dim', type=int, default=64, help='num of encoder layers')
    args.add_argument('--gat_edge_dim', type=int, default=3, help='num of decoder layers')
    args.add_argument('--gat_embed_dim', type=int, default=64, help='dimension of fcn')
    args.add_argument('--gat_node_num', type=int, default=35, help='dimension of fcn')

    args.add_argument('--n_heads', type=int, default=8, help='num of heads')
    args.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    args.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    args.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    args.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    args.add_argument('--factor', type=int, default=1, help='attn factor')
    args.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    args.add_argument('--dropout', type=float, default=0.05, help='dropout')
    args.add_argument('--embed', type=str, default='learned',
                        help='time features encoding, options:[timeF, fixed, learned]')
    args.add_argument('--activation', type=str, default='gelu', help='activation')
    args.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    args.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # parser.add_argument('--itr', type=int, default=2, help='experiments times')
    # parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--des', type=str, default='test', help='exp description')
    # parser.add_argument('--loss', type=str, default='mse', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # # GPU
    # parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


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
    
    main(args, config)
