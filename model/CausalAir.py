import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.CausalAir_EncDec import CausalAirEncoder
from layers.Embed import DataEmbedding_st_v2, Timestamp_Embedding
import numpy as np
import argparse


class CausalAir(nn.Module):
    """
    CausalAir - Causal Air Quality Prediction Network
    
    A causal discovery and prediction network using hard Gumbel-Softmax sampling 
    to generate discrete 0/1 causal matrices.
    
    Key Features:
    - Station-level causal matrix with discrete 0/1 values [N, N] (using Gumbel-Softmax hard sampling)
    - Station-specific variable-level causal matrix with discrete 0/1 values [N, 7, 56] (using Gumbel-Softmax hard sampling)
    - Straight-Through Estimator (STE) for gradient flow: y_hard - y_soft.detach() + y_soft
    - Backdoor replacement formula: O = (A .* C) @ V + (A .* (1-C)) @ Z
      where C is discrete causal matrix (0 or 1)
    - Configurable temperature parameter to control Gumbel-Softmax sampling
    - Dual sparsity regularization for both station-level and variable-level matrices
    - Only station-level attention uses backdoor replacement, variate-level remains unchanged
    
    Temperature Control:
    - Lower temperature (e.g., 0.1): More deterministic, sharper 0/1 distinction
    - Higher temperature (e.g., 1.0): More random, softer gradients during training
    - As training progresses, temperature can be annealed for better discretization
    """

    def __init__(self, configs, **kwargs):
        super(CausalAir, self).__init__()
        configs = argparse.Namespace(**configs)
        self.device = torch.device("cuda")
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.use_fore = configs.use_fore
        self.use_mete = configs.use_mete

        self.time_c = configs.time_c
        self.d_model = configs.d_model
        self.aq_features = configs.gat_node_features
        self.mete_features = configs.mete_features
        self.enc_in = self.aq_features + self.mete_features
        self.n_station = configs.n_station
        self.num_stations = getattr(configs, 'num_stations', self.n_station)
        
        # Noise configuration parameters for backdoor replacement
        backdoor_config = getattr(configs, 'backdoor_attention', {})
        self.noise_type = backdoor_config.get('noise_type', 'gaussian')
        self.noise_std = backdoor_config.get('noise_std', 0.1)
        self.noise_mean = backdoor_config.get('noise_mean', 0.0)
        self.noise_seed = backdoor_config.get('noise_seed', 42)
        
        # Temperature control parameters for Gumbel-Softmax
        self.temperature = backdoor_config.get('temperature', 1.0)
        self.current_temperature = self.temperature  # Track current temperature for annealing
        
        # Add noise and temperature parameters to configs so EncoderLayer can access them
        configs.noise_type = self.noise_type
        configs.noise_std = self.noise_std
        configs.noise_mean = self.noise_mean
        configs.noise_seed = self.noise_seed
        configs.temperature = self.temperature
        configs.num_stations = self.num_stations
        
        # Embedding
        self.enc_embedding = DataEmbedding_st_v2(configs.seq_len, self.pred_len, self.enc_in, configs.d_model, self.aq_features, configs.embed, configs.freq, configs.dropout)
        
        self.timestamp_embedding = Timestamp_Embedding(configs.time_c, configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Learnable logits for station-level causality
        # These are the raw unnormalized logits that will be fed into Gumbel-Softmax
        self.theta = nn.Parameter(torch.randn(self.n_station, self.n_station, 2), requires_grad=True)
        nn.init.xavier_uniform_(self.theta)
        
        # Learnable logits for station-specific variable-level causal relationship
        # Shape: [N, 7, 56, 2] where last dim is [no_edge, edge] logits
        self.station_var_theta = nn.Parameter(torch.randn(self.num_stations, self.aq_features, self.mete_features, 2), requires_grad=True)
        nn.init.xavier_uniform_(self.station_var_theta)
        
        # Sparsity regularization weight
        self.sparsity_weight = getattr(configs, 'sparsity_weight', 0.01)

        self.encoder = CausalAirEncoder(configs, self.enc_in, self.d_model, self.aq_features, self.mete_features)

        # Projector for the initial prediction
        self.initial_projector = nn.Sequential(nn.Linear(configs.d_model, configs.d_model, bias=True),
                                                nn.GELU(),
                                                nn.Dropout(p=configs.dropout),
                                                nn.Linear(configs.d_model, self.pred_len, bias=True))

        # Projector for the final prediction from the future representation
        self.future_projector = nn.Sequential(nn.Linear(configs.d_model, configs.d_model, bias=True),
                                              nn.GELU(),
                                              nn.Dropout(p=configs.dropout),
                                              nn.Linear(configs.d_model, self.pred_len, bias=True))

        # Projector for reconstruction from the historical representation
        self.reconstruct_projector = nn.Sequential(nn.Linear(configs.d_model, configs.d_model, bias=True),
                                                   nn.GELU(),
                                                   nn.Dropout(p=configs.dropout),
                                                   nn.Linear(configs.d_model, self.seq_len, bias=True))

    def gumbel_softmax_binary(self, logits, tau=1.0, hard=True, dim=-1):
        """
        Binary Gumbel-Softmax sampling for discrete 0/1 values
        
        Args:
            logits: [..., 2] unnormalized log probabilities for [no_edge, edge]
            tau: temperature parameter
            hard: if True, return discrete 0/1 values with straight-through gradients
            dim: dimension to apply softmax
            
        Returns:
            Sampled values: if hard=True, returns 0 or 1; else returns soft probabilities
        """
        # Generate Gumbel noise
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        
        # Add Gumbel noise and apply temperature
        gumbels = (logits + gumbels) / tau
        y_soft = F.softmax(gumbels, dim=dim)
        
        if hard:
            # Straight-through estimator
            # Get the index of max value (0 or 1)
            index = y_soft.max(dim, keepdim=True)[1]
            # Create one-hot encoding
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            # Apply straight-through: forward pass uses y_hard, backward pass uses y_soft
            ret = y_hard - y_soft.detach() + y_soft
            # Extract the probability of edge (index 1)
            return ret[..., 1]
        else:
            # Return soft probability of edge
            return y_soft[..., 1]

    def get_station_causal_matrix(self):
        """
        Generate station-level discrete causal matrix using hard Gumbel-Softmax sampling
        
        For station-level causal relationships:
        - causal_matrix[i,j] = 1: station i trusts station j (causal edge exists)
        - causal_matrix[i,j] = 0: station i doesn't trust station j (no causal edge)
        - Used in formula: O = (A .* C) @ V + (A .* (1-C)) @ Z
        
        Returns:
            causal_matrix: [N, N] discrete 0/1 station-level causal matrix
        """
        station_causal_matrix = self.gumbel_softmax_binary(
            self.theta, 
            tau=self.current_temperature,  # Use current_temperature for annealing
            hard=True, 
            dim=-1
        )
        return station_causal_matrix

    def get_station_specific_variable_causal_matrix(self):
        """
        Generate station-specific variable-level discrete causal matrix using hard Gumbel-Softmax sampling
        
        For station-specific variable-level causal relationships (AQ-Mete):
        - causal_matrix[n,i,j] = 1: station n's AQ feature i has causal edge to Mete feature j
        - causal_matrix[n,i,j] = 0: station n's AQ feature i has no causal edge to Mete feature j
        - Used in formula: O = (A .* C) @ V + (A .* (1-C)) @ Z
        
        Returns:
            causal_matrix: [N, 7, 56] discrete 0/1 station-specific variable-level causal matrix
        """
        station_var_causal_matrix = self.gumbel_softmax_binary(
            self.station_var_theta, 
            tau=self.current_temperature,  # Use current_temperature for annealing
            hard=True, 
            dim=-1
        )
        return station_var_causal_matrix

    def compute_sparsity_loss(self, station_causal_matrix, station_var_causal_matrix):
        """
        Compute L1 regularization loss for sparsity
        This encourages both causal matrices to be sparse (most elements close to zero)
        
        Args:
            station_causal_matrix: [N, N] discrete station-level causal matrix
            station_var_causal_matrix: [N, 7, 56] discrete station-specific variable-level causal matrix
            
        Returns:
            sparsity_loss: Combined sparsity loss for both matrices
        """
        station_sparsity = torch.mean(torch.abs(station_causal_matrix))
        station_var_sparsity = torch.mean(torch.abs(station_var_causal_matrix))
        
        return self.sparsity_weight * (station_sparsity + station_var_sparsity)

    def update_temperature(self, epoch, total_epochs, scheduler_config=None):
        """
        Update temperature using annealing strategy
        
        Args:
            epoch: Current training epoch (0-based)
            total_epochs: Total number of training epochs
            scheduler_config: Dictionary with annealing configuration
                - initial_temp: Starting temperature (default: 1.0)
                - final_temp: Ending temperature (default: 0.1)
                - decay_epochs: Number of epochs for annealing (default: total_epochs)
                - decay_type: 'linear' or 'exponential' (default: 'exponential')
        """
        if scheduler_config is None:
            # Default: exponential decay from 1.0 to 0.1 over all epochs
            scheduler_config = {
                'initial_temp': 1.0,
                'final_temp': 0.1,
                'decay_epochs': total_epochs,
                'decay_type': 'exponential'
            }
        
        initial_temp = scheduler_config.get('initial_temp', 1.0)
        final_temp = scheduler_config.get('final_temp', 0.1)
        decay_epochs = scheduler_config.get('decay_epochs', total_epochs)
        decay_type = scheduler_config.get('decay_type', 'exponential')
        
        # Clamp epoch to decay range
        progress = min(epoch / max(decay_epochs, 1), 1.0)
        
        if decay_type == 'exponential':
            # Exponential decay: τ(t) = τ_final + (τ_initial - τ_final) * exp(-5 * progress)
            # At progress=0: τ = τ_initial
            # At progress=1: τ ≈ τ_final
            decay_rate = 5.0  # Controls steepness of decay
            self.current_temperature = final_temp + (initial_temp - final_temp) * np.exp(-decay_rate * progress)
        elif decay_type == 'linear':
            # Linear decay: τ(t) = τ_initial - (τ_initial - τ_final) * progress
            self.current_temperature = initial_temp - (initial_temp - final_temp) * progress
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}. Use 'linear' or 'exponential'.")
        
        # Update temperature in config for consistency
        self.temperature = self.current_temperature
        
        return self.current_temperature

    def get_current_temperature(self):
        """
        Get current temperature value
        
        Returns:
            current_temperature: Current temperature for Gumbel-Softmax sampling
        """
        return self.current_temperature

    def forecast(self, aq_data, mete_data, fore_mete_data, coordinate, time_stamp=None):
        B, NS, L, C = aq_data.shape
        
        # 1. Create historical representations
        aq_x_hist, mete_x_hist = self.enc_embedding(aq_data, mete_data, coordinate, is_future=False)

        # 2. Generate initial prediction from historical AQ representation
        aq_x_hist_flat = aq_x_hist.view(B*NS, C, self.d_model)
        initial_pred_flat = self.initial_projector(aq_x_hist_flat).permute(0, 2, 1)  # [B*NS, pred_len, C]
        initial_pred_for_embed = initial_pred_flat.view(B, NS, self.pred_len, C)
        
        # 3. Create future AQ representation from the initial raw prediction
        future_aq_x, fore_mete_x = self.enc_embedding(initial_pred_for_embed, fore_mete_data, coordinate, is_future=True)

        # 4. Generate discrete causal matrices with hard Gumbel-Softmax sampling
        station_causal_matrix = self.get_station_causal_matrix()
        station_var_causal_matrix = self.get_station_specific_variable_causal_matrix()

        # 5. Encoder pass with both discrete causal matrices
        aq_x_out, _, future_aq_x_out, _, attns = self.encoder(
            aq_x_hist, mete_x_hist, future_aq_x, fore_mete_x, embed_timestamp=None, 
            causal_matrix=station_causal_matrix, station_var_causal_matrix=station_var_causal_matrix
        )
        B, N , C, D= aq_x_out.size()
        # 6. Project for final prediction and reconstruction
        final_prediction = self.future_projector(future_aq_x_out.view(B*N,C,D)).permute(0, 2, 1)
        reconstructed_out = self.reconstruct_projector(aq_x_out.view(B*N,C,D)).permute(0, 2, 1)

        # 7. Compute sparsity loss for both station-level and station-specific variable-level matrices
        sparsity_loss = self.compute_sparsity_loss(station_causal_matrix, station_var_causal_matrix)

        return final_prediction, reconstructed_out, initial_pred_flat, sparsity_loss

    def forward(self, Data, mask=None):
        AQStation_coordinate = Data['AQStation_coordinate'].to(self.device)

        # Prepare historical data
        aq_data = Data['aq_train_data'][:, :self.seq_len, :, -self.aq_features:].to(self.device).transpose(1, 2)
        mete_data = Data['mete_train_data'][:, :self.seq_len, :, :].to(self.device).transpose(1, 2)
        
        # Prepare future forecast data
        fore_data = Data['mete_train_data'][:, self.seq_len:, :, :].to(self.device).transpose(1, 2)
        
        # Prepare timestamp
        time_stamp = Data['aq_train_data'][:, :self.seq_len, 0, 1:self.time_c].to(self.device)
        
        final_pred, reconstructed_out, initial_pred, sparsity_loss = self.forecast(
            aq_data, mete_data, fore_data, AQStation_coordinate, time_stamp
        )

        return final_pred, reconstructed_out, initial_pred, sparsity_loss
