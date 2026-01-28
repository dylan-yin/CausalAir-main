import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

from utils.dist import synchronize, is_main_process, get_rank
from tqdm import tqdm
import os

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, args, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, output_attn=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config, args)
        self.config = config
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.output_attn = output_attn
        self.c_out = args.c_out
        
        # Temperature scheduler configuration for Gumbel-Softmax annealing
        # Use try-except to handle missing temperature_scheduler in config
        try:
            self.temperature_scheduler_config = config['temperature_scheduler']
            self.use_temperature_annealing = True
        except KeyError:
            self.temperature_scheduler_config = None
            self.use_temperature_annealing = False
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            # self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        
        
        self.lr_scheduler = lr_scheduler
        
        # Log temperature scheduler info
        if self.use_temperature_annealing:
            temp_args = self.temperature_scheduler_config['args']
            self.logger.info(f"Temperature annealing enabled: {temp_args['initial_temp']} -> {temp_args['final_temp']} over {temp_args['decay_epochs']} epochs ({temp_args['decay_type']})")
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        if hasattr(self.valid_data_loader.dataset, 'metedata'):
            if self.data_loader.dataset.scale:
                L, N, C = self.valid_data_loader.dataset.metedata.shape
                metedata = self.valid_data_loader.dataset.metedata.reshape(L * N, C)
                metedata = self.data_loader.dataset.mete_scaler.transform(metedata)
                self.valid_data_loader.dataset.metedata = metedata.reshape(L, N, C)
        if hasattr(self.valid_data_loader.dataset, 'AQdata'):
            if self.data_loader.dataset.scale:
                L, N, C = self.valid_data_loader.dataset.AQdata.shape
                AQdata = self.valid_data_loader.dataset.AQdata.reshape(L * N, C)
                AQdata[:, -7:] = self.data_loader.dataset.aq_scaler.transform(AQdata[:, -7:])
                self.valid_data_loader.dataset.AQdata = AQdata.reshape(L, N, C)

        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_mean = torch.Tensor(self.data_loader.dataset.aq_scaler.mean_).to(self.device)
        self.val_var = torch.sqrt(torch.Tensor(self.data_loader.dataset.aq_scaler.var_).to(self.device))
        self.loss_weight = torch.tensor([1,0.7,0.7,0.5,0.5,0.5,0.5]).to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # Update temperature for Gumbel-Softmax annealing (CUTS+ style)
        if self.use_temperature_annealing and hasattr(self.model, 'update_temperature'):
            temp_config = self.temperature_scheduler_config['args']
            current_temp = self.model.update_temperature(
                epoch=epoch - 1,  # epoch is 1-based, convert to 0-based
                total_epochs=self.epochs,
                scheduler_config=temp_config
            )
            # Log current temperature
            if is_main_process():
                self.logger.info(f'Epoch {epoch}: Temperature = {current_temp:.6f}')
                self.writer.add_scalar('temperature', current_temp, epoch)
        # outputs, targets = self._valid_epoch(epoch)
        pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, (data, target) in pbar:
            
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(self.device)
            for key, value in target.items():
                if torch.is_tensor(value):
                    target[key] = value.to(self.device)
            # target = target.to(self.device)

            self.optimizer.zero_grad()
            
            model_outputs = self.model(data)
            if len(model_outputs) == 5:
                # STformer_v9_9 with dynamic causal matrices
                output, reconstructed_out, initial_pred, total_sparsity_loss, dynamic_causal_matrices = model_outputs
                prediction_loss = self.criterion(output, target['label'][:,:,:self.c_out])
                reconstruction_loss = self.criterion(reconstructed_out, target['reconstructed_label'][:,:,:self.c_out])
                initial_loss = self.criterion(initial_pred, target['label'][:,:,:self.c_out])
                loss = prediction_loss + reconstruction_loss + initial_loss + total_sparsity_loss
            elif len(model_outputs) == 4:
                # STformer_v9_6/v9_8 with sparsity loss
                output, reconstructed_out, initial_pred, sparsity_loss = model_outputs
                prediction_loss = self.criterion(output, target['label'][:,:,:self.c_out])
                reconstruction_loss = self.criterion(reconstructed_out, target['reconstructed_label'][:,:,:self.c_out])
                initial_loss = self.criterion(initial_pred, target['label'][:,:,:self.c_out])
                loss = prediction_loss + reconstruction_loss + initial_loss + sparsity_loss
            elif len(model_outputs) == 3:
                output, reconstructed_out, initial_pred = model_outputs
                loss = self.criterion(output, target['label'][:,:,:self.c_out]) + \
                       self.criterion(reconstructed_out, target['reconstructed_label'][:,:,:self.c_out]) + \
                       self.criterion(initial_pred, target['label'][:,:,:self.c_out])
            elif len(model_outputs) == 2:
                # STformer_v9_7 with only prediction and initial prediction (no reconstruction)
                output, initial_pred = model_outputs
                prediction_loss = self.criterion(output, target['label'][:,:,:self.c_out])
                initial_loss = self.criterion(initial_pred, target['label'][:,:,:self.c_out])
                loss = prediction_loss + initial_loss
            else:
                output,reconstructed_out = model_outputs
                loss = self.criterion(output[:, :], target['label'][:,:,:self.c_out]) + self.criterion(reconstructed_out, target['reconstructed_label'][:,:,:self.c_out])

            loss_reduced = self.reduce_loss(loss)
            loss.backward()
            self.optimizer.step()

            if is_main_process():
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss_reduced.item())

                # if batch_idx % self.log_step == 0:
                #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                #         epoch,
                #         self._progress(batch_idx+1),
                #         loss_reduced.item()))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            pbar.set_description('Train Epoch: {} {} '.format(epoch,self._progress(batch_idx+1)))

            pbar.set_postfix(train_loss=loss_reduced.item())
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            self.valid_metrics.reset()
            outputs, targets = self._valid_epoch(epoch)
            synchronize()
            # outputs = self._accumulate_predictions_from_multiple_gpus(outputs)
            # targets = self._accumulate_predictions_from_multiple_gpus(targets)

            if is_main_process():
                # outputs = [output.to(self.device) for output in outputs]
                # targets = [target.to(self.device) for target in targets]
                # loss = self.criterion(output[:, :], target)
                #
                # loss_reduced = self.reduce_loss(loss)
                # self.valid_metrics.update('loss', loss_reduced.item())
                # outputs = torch.cat(outputs,dim=0)*self.val_mean +self.val_mean
                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(torch.cat(outputs,dim=0), torch.cat(targets,dim=0)))
                val_log = self.valid_metrics.result()
                log.update(**{'val_'+k : v for k, v in val_log.items()})

            if self.output_attn:

                if not os.path.exists('attns'):
                    os.makedirs('attns')
                print(os.path.join('attns', 'attns_of_Epoch{}.npy'))
                np.save(os.path.join('attns', 'attns_of_Epoch{}.npy'.format(epoch)), attns.cpu().numpy())

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        outputs = []
        targets = []
        attns = []
        with torch.no_grad():
            pbar = tqdm(enumerate(self.valid_data_loader),
               total=int(len(self.valid_data_loader.sampler) / self.valid_data_loader.batch_size) + 1, leave=False)
            for batch_idx, (data, target) in pbar:

                for key, value in data.items():
                    if torch.is_tensor(value):
                        data[key] = value.to(self.device)
                for key, value in target.items():
                    if torch.is_tensor(value):
                        target[key] = value.to(self.device)
            # target = target.to(self.device)

                model_outputs = self.model(data)
                if len(model_outputs) == 5:
                    # STformer_v9_9/v9_11/v9_18 with dynamic causal matrices
                    output, _, _, total_sparsity_loss, dynamic_causal_matrix = model_outputs
                    
                    # Calculate and update static sparsity (from theta matrix)
                    if hasattr(self.model, 'theta'):
                        prob_matrix = torch.sigmoid(self.model.theta)
                        static_sparsity = torch.sum(torch.abs(prob_matrix)).item()
                        self.valid_metrics.update('Static_L1_Sparsity', static_sparsity)
                    
                    # Calculate and update dynamic sparsity
                    if isinstance(dynamic_causal_matrix, list):
                        # For models that return a list of matrices (like v9_9)
                        dynamic_sparsity = sum(torch.sum(torch.abs(dcm)).item() for dcm in dynamic_causal_matrix)
                    else:
                        # For models that return a single matrix (like v9_11, v9_18)
                        dynamic_sparsity = torch.sum(torch.abs(dynamic_causal_matrix)).item()
                    self.valid_metrics.update('Dynamic_L1_Sparsity', dynamic_sparsity)
                    
                elif len(model_outputs) == 4:
                    # CausalAir with sparsity loss
                    output, _, _, sparsity_loss = model_outputs
                    
                    # Calculate and update static sparsity
                    # For CausalAir: calculate sparsity from discrete causal matrices
                    if hasattr(self.model, 'get_station_causal_matrix') and hasattr(self.model, 'get_station_specific_variable_causal_matrix'):
                        # CausalAir with discrete causal matrices
                        with torch.no_grad():
                            station_causal = self.model.get_station_causal_matrix()
                            station_var_causal = self.model.get_station_specific_variable_causal_matrix()
                            station_sparsity = torch.sum(torch.abs(station_causal)).item()
                            station_var_sparsity = torch.sum(torch.abs(station_var_causal)).item()
                            total_sparsity = station_sparsity + station_var_sparsity
                            self.valid_metrics.update('Static_L1_Sparsity', total_sparsity)
                            
                            # Record current temperature if available
                            if hasattr(self.model, 'get_current_temperature'):
                                current_temp = self.model.get_current_temperature()
                                # Only update once per validation epoch
                                if batch_idx == 0 and is_main_process():
                                    self.writer.add_scalar('validation/temperature', current_temp, epoch)
                    
                    # For other models with theta matrix
                    elif hasattr(self.model, 'theta'):
                        prob_matrix = torch.sigmoid(self.model.theta)
                        static_sparsity = torch.sum(torch.abs(prob_matrix)).item()
                        self.valid_metrics.update('Static_L1_Sparsity', static_sparsity)
                    
                elif len(model_outputs) == 3:
                    output, _, _ = model_outputs
                elif len(model_outputs) == 2:
                    # STformer_v9_7 with only prediction and initial prediction
                    output, _ = model_outputs
                else:
                    output, _ = model_outputs

                output = output*self.val_var + self.val_mean
                target =  target['label']*self.val_var + self.val_mean
                # output = self.data_loader.dataset.aq_scaler.inverse_transform(output)
                # target = self.data_loader.dataset.aq_scaler.inverse_transform(target['label'])
                for met in self.metric_ftns:
                    # Skip sparsity metrics as they are calculated separately
                    if met.__name__ not in ['Static_L1_Sparsity', 'Dynamic_L1_Sparsity']:
                        self.valid_metrics.update(met.__name__, met(output, target))
                if batch_idx == self.len_epoch:
                    break
                # outputs.append(output.clone()[:, :])
                # targets.append(target.clone())
                # attns.append(attn[0])
                # if batch_idx == self.len_epoch:
                #     break
        pbar.set_description('Val Epoch: {} {} '.format(epoch, self._progress(batch_idx + 1)))

        return None,None #torch.cat(outputs,dim=0), torch.cat(targets,dim=0),# torch.cat(attns,dim=0)
    def invert_trans(self,data):
        B,L,C = data.size()
        return self.valid_data_loader.dataset.scaler.inverse_transform(data.reshape(B*L,C)).reshape(B,L,C)


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
