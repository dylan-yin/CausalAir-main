import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, FullAttention_v42, FullAttention_v52, AttentionLayer

class DualAttentionModule(nn.Module):
    """
    Dual Attention Module (DAM) for meteorological feature extraction.
    Acts like an AttentionLayer but uses an internal, learnable token as the query.
    It is designed to distill information from a sequence into a single vector.
    """
    def __init__(self, configs, d_model, n_heads, dropout):
        super(DualAttentionModule, self).__init__()
        self.attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=dropout, output_attention=configs.output_attention), d_model, n_heads)
        self.token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        nn.init.kaiming_normal_(self.token, mode='fan_in')

    def forward(self, key, value, attn_mask=None):
        B_K, _, _ = key.shape
        query = self.token.repeat(B_K, 1, 1)
        updated_token, attn_weights = self.attention(query, key, value, attn_mask=attn_mask)
        return updated_token, attn_weights

class CausalAirEncoderLayer(nn.Module):
    """
    CausalAir Encoder Layer - Causal Air Quality Prediction Network Encoder Layer
    
    Uses discrete 0/1 causal matrices with Gumbel-Softmax sampling.
    
    Key Features:
    - Station-level discrete causal matrix [N, N] with 0/1 values
    - Station-specific variable-level discrete causal matrix [N, 7, 56] with 0/1 values
    - Backdoor replacement formula: O = (A .* C) @ V + (A .* (1-C)) @ Z
      where C is discrete causal matrix (0 or 1)
    - Only station-level attention uses backdoor replacement, variate-level remains unchanged
    - Meteorological updates use station-specific variable-level causal matrix for aq-mete causal relationship
    """
    
    def __init__(self, configs, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(CausalAirEncoderLayer, self).__init__()
        self.device = torch.device("cuda")
        self.strategy = configs.mete_update_strategy
        d_ff = d_ff or 4 * d_model
        self.n_heads = configs.n_heads
        self.num_stations = getattr(configs, 'num_stations', 184)  # Default to 184 stations
        
        # Get noise configuration parameters from configs
        noise_type = getattr(configs, 'noise_type', 'gaussian')
        noise_std = getattr(configs, 'noise_std', 0.1)
        noise_mean = getattr(configs, 'noise_mean', 0.0)
        noise_seed = getattr(configs, 'noise_seed', 42)
        
        # Use FullAttention_v42 for station-level attention (with backdoor replacement)
        self.station_self_attention = AttentionLayer(
            FullAttention_v42(True, configs.factor, attention_dropout=configs.dropout, 
                           output_attention=configs.output_attention,
                           noise_type=noise_type, noise_std=noise_std, 
                           noise_mean=noise_mean, noise_seed=noise_seed), 
            configs.d_model, configs.n_heads)
        
        # Keep FullAttention for variate-level attention (no backdoor replacement)
        self.variate_self_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads)

        # Self-attention for future_aq_x - use FullAttention_v42 for station-level
        self.future_station_self_attention = AttentionLayer(
            FullAttention_v42(True, configs.factor, attention_dropout=configs.dropout, 
                           output_attention=configs.output_attention,
                           noise_type=noise_type, noise_std=noise_std, 
                           noise_mean=noise_mean, noise_seed=noise_seed), 
            configs.d_model, configs.n_heads)
        
        # Keep FullAttention for future variate-level attention (no backdoor replacement)
        self.future_variate_self_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads)

        if self.strategy == 'dam':
            self.hist_mete_updater = DualAttentionModule(configs, d_model, configs.n_heads, dropout)
            self.fore_mete_updater = DualAttentionModule(configs, d_model, configs.n_heads, dropout)
        elif self.strategy == 'cross_attention':
            # Use FullAttention_v52 for meteorological updates with station-specific variable-level causal matrix
            self.hist_mete_updater = AttentionLayer(
                FullAttention_v52(False, configs.factor, attention_dropout=dropout, 
                               output_attention=configs.output_attention,
                               noise_type=noise_type, noise_std=noise_std, 
                               noise_mean=noise_mean, noise_seed=noise_seed), 
                d_model, configs.n_heads)
            self.fore_mete_updater = AttentionLayer(
                FullAttention_v52(False, configs.factor, attention_dropout=dropout, 
                               output_attention=configs.output_attention,
                               noise_type=noise_type, noise_std=noise_std, 
                               noise_mean=noise_mean, noise_seed=noise_seed), 
                d_model, configs.n_heads)
        else:
            raise ValueError(f"Unknown mete_update_strategy: {self.strategy}")

        # Use FullAttention_v42 for station-level cross-attention (with backdoor replacement)
        self.station_cross_attention = AttentionLayer(
            FullAttention_v42(True, configs.factor, attention_dropout=configs.dropout, 
                           output_attention=configs.output_attention,
                           noise_type=noise_type, noise_std=noise_std, 
                           noise_mean=noise_mean, noise_seed=noise_seed), 
            configs.d_model, configs.n_heads)
        
        # Keep FullAttention for variate-level cross-attention (no backdoor replacement)
        self.variate_cross_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads)

        self.conv1_future = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_future = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_future = nn.LayerNorm(d_model)
        self.norm2_future = nn.LayerNorm(d_model)

        self.conv1_aq = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_aq = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_aq = nn.LayerNorm(d_model)
        self.norm2_aq = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def apply_station_causal_mask(self, causal_matrix, batch_size, seq_len):
        """
        Apply station-level discrete causal matrix to attention computation
        For station-level causal relationships:
        - causal_matrix[i,j] = 1: station i trusts station j (causal edge exists)
        - causal_matrix[i,j] = 0: station i doesn't trust station j (no causal edge)
        - Used in formula: O = (A .* C) @ V + (A .* (1-C)) @ Z
        
        Args:
            causal_matrix: [N, N] discrete 0/1 station-level causal matrix
            batch_size: batch size for reshaping
            seq_len: sequence length for reshaping
        Returns:
            causal_matrix: [N, N] tensor directly for attention computation
        """
        if causal_matrix is None:
            return None
        
        return causal_matrix

    def apply_variate_causal_mask(self, causal_matrix, batch_size, num_vars):
        """
        For variate attention, we don't apply station causal matrix
        Returns None to indicate no causal weighting for variate dimension
        """
        return None

    def apply_variable_causal_mask(self, station_var_causal_matrix):
        """
        Apply station-specific variable-level discrete causal matrix for aq-mete causal relationship
        For station-specific variable-level causal relationships (AQ-Mete):
        - causal_matrix[n,i,j] = 1: station n's AQ feature i has causal edge to Mete feature j
        - causal_matrix[n,i,j] = 0: station n's AQ feature i has no causal edge to Mete feature j
        - Used in formula: O = (A .* C) @ V + (A .* (1-C)) @ Z
        
        Args:
            station_var_causal_matrix: [N, 7, 56] discrete 0/1 station-specific variable-level causal matrix from main model
        Returns:
            station_var_causal_matrix: [N, 7, 56] tensor directly for FullAttention_v52
        """
        if station_var_causal_matrix is None:
            return None
        
        return station_var_causal_matrix

    def forward(self, aq_x, mete_x, future_aq_x, fore_mete_x, embed_timestamp, causal_matrix=None, station_var_causal_matrix=None):
        B, N, C, D = aq_x.size()

        # 1. Self-attention for aq_x with station-level discrete causal matrix
        aq_x_n_self = aq_x.permute(0, 2, 1, 3).contiguous().view(B * C, N, D)
        station_causal = self.apply_station_causal_mask(causal_matrix, B * C, N)
        station_self_out, _ = self.station_self_attention(aq_x_n_self, aq_x_n_self, aq_x_n_self, attn_mask=station_causal)
        station_self_out = station_self_out.view(B, C, N, D).permute(0, 2, 1, 3)
        
        aq_x_c_self = aq_x.view(B * N, C, D)
        variate_causal = self.apply_variate_causal_mask(causal_matrix, B * N, C)
        variate_self_out, _ = self.variate_self_attention(aq_x_c_self, aq_x_c_self, aq_x_c_self, attn_mask=variate_causal)
        variate_self_out = variate_self_out.view(B, N, C, D)
        aq_x = aq_x + self.dropout(station_self_out) + self.dropout(variate_self_out)

        mete_x_flat = mete_x.view(B * N, -1, D)
        fore_mete_x_flat = fore_mete_x.view(B * N, -1, D)
        
        # Get station-specific variable-level discrete causal matrix for meteorological updates
        station_var_causal = self.apply_variable_causal_mask(station_var_causal_matrix) if station_var_causal_matrix is not None else None
        
        # 2. Meteorological update for aq_x with station-specific variable-level discrete causal matrix
        if self.strategy == 'dam':
            mete_token, _ = self.hist_mete_updater(mete_x_flat, mete_x_flat, attn_mask=None)
            aq_x = aq_x + mete_token.view(B, N, 1, D)
        elif self.strategy == 'cross_attention':
            aq_x_flat = aq_x.view(B*N, C, D)
            hist_update, _ = self.hist_mete_updater(aq_x_flat, mete_x_flat, mete_x_flat, attn_mask=station_var_causal)
            aq_x = aq_x + self.dropout(hist_update.view(B, N, C, D))

        # 3. Cross-attention for future_aq_x with aq_x (using station-level discrete causal matrix)
        future_aq_x_n = future_aq_x.permute(0, 2, 1, 3).contiguous().view(B * C, N, D)
        aq_x_n_cross = aq_x.permute(0, 2, 1, 3).contiguous().view(B * C, N, D)
        station_update, _ = self.station_cross_attention(future_aq_x_n, aq_x_n_cross, aq_x_n_cross, attn_mask=station_causal)
        station_update = station_update.view(B, C, N, D).permute(0, 2, 1, 3)

        future_aq_x_c = future_aq_x.view(B * N, C, D)
        aq_x_c_cross = aq_x.view(B * N, C, D)
        variate_update, _ = self.variate_cross_attention(future_aq_x_c, aq_x_c_cross, aq_x_c_cross, attn_mask=None)
        variate_update = variate_update.view(B, N, C, D)
        future_aq_x = future_aq_x + self.dropout(station_update) + self.dropout(variate_update)
        
        # 4. Meteorological update for future_aq_x with station-specific variable-level discrete causal matrix
        if self.strategy == 'dam':
            fore_mete_token, _ = self.fore_mete_updater(fore_mete_x_flat, fore_mete_x_flat, attn_mask=None)
            future_aq_x = future_aq_x + fore_mete_token.view(B, N, 1, D)
        elif self.strategy == 'cross_attention':
            future_aq_x_flat = future_aq_x.view(B*N, C, D)
            future_update, _ = self.fore_mete_updater(future_aq_x_flat, fore_mete_x_flat, fore_mete_x_flat, attn_mask=station_var_causal)
            future_aq_x = future_aq_x + self.dropout(future_update.view(B, N, C, D))

        # 5. Self-attention for future_aq_x (using station-level discrete causal matrix)
        future_aq_x_n_self = future_aq_x.permute(0, 2, 1, 3).contiguous().view(B * C, N, D)
        future_station_self_out, _ = self.future_station_self_attention(future_aq_x_n_self, future_aq_x_n_self, future_aq_x_n_self, attn_mask=station_causal)
        future_station_self_out = future_station_self_out.view(B, C, N, D).permute(0, 2, 1, 3)

        future_aq_x_c_self = future_aq_x.view(B * N, C, D)
        future_variate_self_out, _ = self.future_variate_self_attention(future_aq_x_c_self, future_aq_x_c_self, future_aq_x_c_self, attn_mask=None)
        future_variate_self_out = future_variate_self_out.view(B, N, C, D)
        future_aq_x = future_aq_x + self.dropout(future_station_self_out) + self.dropout(future_variate_self_out)

        def ffn_future(x_in):
            y = self.norm1_future(x_in)
            y = self.dropout(self.activation(self.conv1_future(y.transpose(-1, 1))))
            y = self.dropout(self.conv2_future(y).transpose(-1, 1))
            return self.norm2_future(x_in + y)
        
        def ffn_aq(x_in):
            y = self.norm1_aq(x_in)
            y = self.dropout(self.activation(self.conv1_aq(y.transpose(-1, 1))))
            y = self.dropout(self.conv2_aq(y).transpose(-1, 1))
            return self.norm2_aq(x_in + y)

        aq_x = ffn_aq(aq_x.view(B*N, C, D)).view(B, N, C, D)
        future_aq_x = ffn_future(future_aq_x.view(B*N, C, D)).view(B, N, C, D)

        return aq_x, mete_x, future_aq_x, fore_mete_x, None


class CausalAirEncoder(nn.Module):
    """
    CausalAir Encoder - Causal Air Quality Prediction Network Encoder
    
    Uses discrete 0/1 causal matrices with Gumbel-Softmax sampling.
    
    Key Features:
    - Station-level discrete causal matrix [N, N] with 0/1 values
    - Station-specific variable-level discrete causal matrix [N, 7, 56] with 0/1 values
    - Dual causal matrix support for comprehensive causal modeling
    - Flexible handling of missing causal matrices (graceful degradation)
    """
    
    def __init__(self, configs, enc_in ,d_model, aq_features,mete_features, ):
        super(CausalAirEncoder, self).__init__()
        self.attn_layers = nn.ModuleList([
                CausalAirEncoderLayer( configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ])
        self.norm = torch.nn.LayerNorm(configs.d_model)

    def forward(self, aq_x, mete_x, future_aq_x, fore_mete_x, embed_timestamp=None, causal_matrix=None, station_var_causal_matrix=None):
        """
        Forward pass with dual discrete causal matrix support
        
        Args:
            aq_x: AQ historical data [B, N, C, D]
            mete_x: Meteorological historical data [B, N, M, D]
            future_aq_x: Future AQ data [B, N, C, D]
            fore_mete_x: Future meteorological data [B, N, M, D]
            embed_timestamp: Timestamp embeddings (optional)
            causal_matrix: Station-level discrete causal matrix [N, N] with 0/1 values (optional)
            station_var_causal_matrix: Station-specific variable-level discrete causal matrix [N, 7, 56] with 0/1 values (optional)
            
        Returns:
            aq_x: Updated AQ historical representation
            mete_x: Meteorological historical data (unchanged)
            future_aq_x: Updated future AQ representation
            fore_mete_x: Future meteorological data (unchanged)
            attns: Attention weights (list)
        """
        attns = []
        for attn_layer in self.attn_layers:
            aq_x, mete_x, future_aq_x, fore_mete_x, attn = attn_layer(
                aq_x, mete_x, future_aq_x, fore_mete_x, embed_timestamp, 
                causal_matrix=causal_matrix, station_var_causal_matrix=station_var_causal_matrix
            )
            attns.append(attn)

        if self.norm is not None:
            aq_x = self.norm(aq_x)
            future_aq_x = self.norm(future_aq_x)

        return aq_x, mete_x, future_aq_x, fore_mete_x, attns
