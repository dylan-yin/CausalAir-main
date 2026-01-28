import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
import os


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            # Return the original attention weights A for meteorological attention extraction
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention_v43(nn.Module):
    """
    FullAttention_v43: Modified version of FullAttention_v4 with simplified backdoor replacement logic.
    Instead of generating noise and replacing values, directly applies causal matrix as attention mask.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_v43, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _validate_prob_matrix(self, prob_matrix, expected_shape):
        """
        Validate prob_matrix dimensions and values.
        
        Args:
            prob_matrix: Probability matrix to validate
            expected_shape: Expected shape tuple (N, N)
            
        Returns:
            validated_prob_matrix: Validated and potentially corrected prob_matrix
        """
        if prob_matrix is None:
            raise ValueError(f" prob_matrix is None ") 
            
        # Check dimensions
        if len(prob_matrix.shape) != 2:
            raise ValueError(f"prob_matrix must be 2D, got shape: {prob_matrix.shape}")
            
        if prob_matrix.shape != expected_shape:
            raise ValueError(f"prob_matrix shape {prob_matrix.shape} doesn't match "
                           f"expected shape {expected_shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(prob_matrix).any() or torch.isinf(prob_matrix).any():
            print("Warning: prob_matrix contains NaN or infinite values, clamping to [0, 1]")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        # Ensure values are in [0, 1] range
        if (prob_matrix < 0).any() or (prob_matrix > 1).any():
            print("Warning: prob_matrix values outside [0, 1] range, clamping")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        return prob_matrix

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores: Q @ K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Use attn_mask directly as causal matrix (prob_matrix)
        causal_matrix = attn_mask  # Expected shape: [N, N] or [B, H, N, N]
        
        # Handle different causal_matrix shapes
        if len(causal_matrix.shape) == 2:
            # causal_matrix is [N, N], need to expand to [B, H, N, N]
            N1, N2 = causal_matrix.shape
            if N1 != L or N2 != S:
                raise ValueError(f"causal_matrix shape [{N1}, {N2}] doesn't match "
                                f"sequence lengths [L={L}, S={S}]")
            
            # Validate causal_matrix
            causal_matrix = self._validate_prob_matrix(causal_matrix, (N1, N2))
            
            # Expand to batch and head dimensions: [N, N] -> [B, H, N, N]
            C = causal_matrix.unsqueeze(0).unsqueeze(0).expand(B, H, L, S)
            
        else:
            raise ValueError(f"causal_matrix must be 2D [N, N] or 4D [B, H, N, N], "
                            f"got shape: {causal_matrix.shape}")
        
        # Ensure C is on the same device as other tensors
        if C.device != scores.device:
            C = C.to(scores.device)
        
        # Apply causal matrix as attention mask: multiply scores with causal matrix
        # This directly modulates the attention weights based on the causal relationships
        masked_scores = scores * C
        
        # Apply scaling and softmax to get attention weights A
        A = self.dropout(torch.softmax(scale * masked_scores, dim=-1))
        
        # Standard attention computation: A @ V
        V = torch.einsum("bhls,bshd->blhd", A, values)
            
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention_v53(nn.Module):
    """
    FullAttention_v53: Modified version of FullAttention_v5 with simplified backdoor replacement logic.
    Instead of generating noise and replacing values, directly applies causal matrix as attention mask.
    Supports station-specific variable-level causal matrix with shape [N, 7, 56].
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_v53, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _validate_station_specific_causal_matrix(self, causal_matrix, expected_shape):
        """
        Validate station-specific causal_matrix dimensions and values.
        
        Args:
            causal_matrix: Causal matrix to validate [N, 7, 56]
            expected_shape: Expected shape tuple (N, 7, 56)
            
        Returns:
            validated_causal_matrix: Validated and potentially corrected causal_matrix
        """
        if causal_matrix is None:
            raise ValueError(f" prob_matrix is None ") 
            
        # Check dimensions
        if len(causal_matrix.shape) != 3:
            raise ValueError(f"Station-specific causal_matrix must be 3D [N, 7, 56], got shape: {causal_matrix.shape}")
            
        if causal_matrix.shape != expected_shape:
            raise ValueError(f"causal_matrix shape {causal_matrix.shape} doesn't match "
                           f"expected shape {expected_shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(causal_matrix).any() or torch.isinf(causal_matrix).any():
            print("Warning: causal_matrix contains NaN or infinite values, clamping to [0, 1]")
            causal_matrix = torch.clamp(causal_matrix, 0.0, 1.0)
            
        # Ensure values are in [0, 1] range
        if (causal_matrix < 0).any() or (causal_matrix > 1).any():
            print("Warning: causal_matrix values outside [0, 1] range, clamping")
            causal_matrix = torch.clamp(causal_matrix, 0.0, 1.0)
            
        return causal_matrix

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores: Q @ K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Use attn_mask directly as causal matrix
        causal_matrix = attn_mask  # Expected shape: [N, 7, 56]
        
        # Handle station-specific causal_matrix [N, 7, 56]
        if len(causal_matrix.shape) == 3:
            N, aq_features, mete_features = causal_matrix.shape
            
            # Validate causal_matrix
            causal_matrix = self._validate_station_specific_causal_matrix(causal_matrix, (N, aq_features, mete_features))
            
            # For station-specific processing, we need to handle each station independently
            # Input shapes: queries [B*N, L, H, E], keys [B*N, S, H, E], values [B*N, S, H, D]
            # where B*N represents batch_size * num_stations
            
            if B % N != 0:
                raise ValueError(f"Batch size {B} is not divisible by number of stations {N}")
            
            actual_batch_size = B // N
            
            # Reshape scores to separate batch and station dimensions
            # [B*N, H, L, S] -> [actual_batch_size, N, H, L, S]
            scores_reshaped = scores.view(actual_batch_size, N, H, L, S)
            
            # Expand causal_matrix to match attention dimensions
            # [N, 7, 56] -> [actual_batch_size, N, H, 7, 56]
            C = causal_matrix.unsqueeze(0).unsqueeze(2).expand(actual_batch_size, N, H, aq_features, mete_features)
            
            # Ensure C is on the same device as other tensors
            if C.device != scores.device:
                C = C.to(scores.device)
            
            # Apply causal matrix as attention mask: multiply scores with causal matrix
            # This directly modulates the attention weights based on the causal relationships
            masked_scores = scores_reshaped * C  # [actual_batch_size, N, H, L, S]
            
            # Reshape back to original format: [actual_batch_size, N, H, L, S] -> [B, H, L, S]
            masked_scores = masked_scores.view(B, H, L, S)
                
        else:
            raise ValueError(f"FullAttention_v53 expects causal_matrix with shape [N, 7, 56], "
                            f"got shape: {causal_matrix.shape}")
        
        # Apply scaling and softmax to get attention weights A
        A = self.dropout(torch.softmax(scale * masked_scores, dim=-1))
        
        # Standard attention computation: A @ V
        V = torch.einsum("bhls,bshd->blhd", A, values)
            
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class FullAttention_v2(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_v2, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 将scores和attn_mask做按位乘法
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            A = A * attn_mask.mask

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class FullAttention_v3(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_v3, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # For v9.12: directly multiply prob_matrix with attention scores before softmax
        if self.mask_flag and attn_mask is not None and hasattr(attn_mask, 'prob_matrix'):
            # attn_mask.prob_matrix shape: [B, H, L, S]
            scores = scores * attn_mask.prob_matrix

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_Q, L_V]) / L_V).type_as(attn).to(attn.device)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class FullAttention_v4(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 noise_type='gaussian', noise_std=0.1, noise_mean=0.0, noise_seed=42):
        super(FullAttention_v4, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Noise configuration parameters
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise_seed = noise_seed
        
        # Set random seed for reproducibility
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)

    def generate_noise(self, shape, device, dtype):
        """
        Generate configurable noise tensor matching the shape and device of input tensor.
        
        Args:
            shape: Target tensor shape
            device: Target device (cuda/cpu)
            dtype: Target data type
            
        Returns:
            noise_tensor: Generated noise tensor with specified configuration
        """
        # Set seed for reproducible noise generation
        if self.noise_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.noise_seed)
        else:
            generator = None
            
        if self.noise_type == 'gaussian':
            # Generate Gaussian noise with specified mean and std
            if generator is not None:
                noise = torch.normal(mean=self.noise_mean, std=self.noise_std, 
                                   size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.normal(mean=self.noise_mean, std=self.noise_std, 
                                   size=shape, device=device, dtype=dtype)
                                   
        elif self.noise_type == 'uniform':
            # Generate uniform noise in range [mean-std, mean+std]
            low = self.noise_mean - self.noise_std
            high = self.noise_mean + self.noise_std
            if generator is not None:
                noise = torch.rand(size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.rand(size=shape, device=device, dtype=dtype)
            noise = noise * (high - low) + low
            
        elif self.noise_type == 'zero':
            # Generate zero noise (no replacement effect)
            noise = torch.zeros(size=shape, device=device, dtype=dtype)
            
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}. "
                           f"Supported types: 'gaussian', 'uniform', 'zero'")
        
        return noise

    def _validate_prob_matrix(self, prob_matrix, expected_shape):
        """
        Validate prob_matrix dimensions and values.
        
        Args:
            prob_matrix: Probability matrix to validate
            expected_shape: Expected shape tuple (N, N)
            
        Returns:
            validated_prob_matrix: Validated and potentially corrected prob_matrix
        """
        if prob_matrix is None:
            raise ValueError(f" prob_matrix is None ") 
            
        # Check dimensions
        if len(prob_matrix.shape) != 2:
            raise ValueError(f"prob_matrix must be 2D, got shape: {prob_matrix.shape}")
            
        if prob_matrix.shape != expected_shape:
            raise ValueError(f"prob_matrix shape {prob_matrix.shape} doesn't match "
                           f"expected shape {expected_shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(prob_matrix).any() or torch.isinf(prob_matrix).any():
            print("Warning: prob_matrix contains NaN or infinite values, clamping to [0, 1]")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        # Ensure values are in [0, 1] range
        if (prob_matrix < 0).any() or (prob_matrix > 1).any():
            print("Warning: prob_matrix values outside [0, 1] range, clamping")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        return prob_matrix

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores: Q @ K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Apply scaling and softmax to get attention weights A
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
 
        # Use attn_mask directly as prob_matrix tensor
        prob_matrix = attn_mask  # Expected shape: [N, N] or [B, H, N, N]
        
        # Handle different prob_matrix shapes
        if len(prob_matrix.shape) == 2:
            # prob_matrix is [N, N], need to expand to [B, H, N, N]
            N1, N2 = prob_matrix.shape
            if N1 != L or N2 != S:
                raise ValueError(f"prob_matrix shape [{N1}, {N2}] doesn't match "
                                f"sequence lengths [L={L}, S={S}]")
            
            # Validate prob_matrix
            prob_matrix = self._validate_prob_matrix(prob_matrix, (N1, N2))
            
            # Expand to batch and head dimensions: [N, N] -> [B, H, N, N]
            P = prob_matrix.unsqueeze(0).unsqueeze(0).expand(B, H, L, S)
            
        else:
            raise ValueError(f"prob_matrix must be 2D [N, N] or 4D [B, H, N, N], "
                            f"got shape: {prob_matrix.shape}")
        
        # Ensure P is on the same device as other tensors
        if P.device != values.device:
            P = P.to(values.device)
        
        # Generate noise tensor Z with same shape as values
        Z = self.generate_noise(values.shape, values.device, values.dtype)
        
        # Apply backdoor replacement using matrix operations: O = (A .* P) @ V + (A .* (1-P)) @ Z
        # A .* P: element-wise multiplication of attention weights with prob_matrix
        weighted_attention_v = A * P  # [B, H, L, S]
        weighted_attention_z = A * (1.0 - P)  # [B, H, L, S]
        
        # Matrix multiplication: [B, H, L, S] @ [B, S, H, D] -> [B, L, H, D]
        V_original = torch.einsum("bhls,bshd->blhd", weighted_attention_v, values)
        V_noise = torch.einsum("bhls,bshd->blhd", weighted_attention_z, Z)
        
        # Final output: combination of original and noise components
        V = V_original + V_noise
            
       

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention_v5(nn.Module):
    """
    FullAttention_v5: Enhanced version supporting station-specific variable-level prob_matrix
    Supports prob_matrix with shape [N, 7, 56] where N is number of stations,
    7 is aq features, 56 is mete features
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 noise_type='gaussian', noise_std=0.1, noise_mean=0.0, noise_seed=42):
        super(FullAttention_v5, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Noise configuration parameters
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise_seed = noise_seed
        
        # Set random seed for reproducibility
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)

    def generate_noise(self, shape, device, dtype):
        """
        Generate configurable noise tensor matching the shape and device of input tensor.
        
        Args:
            shape: Target tensor shape
            device: Target device (cuda/cpu)
            dtype: Target data type
            
        Returns:
            noise_tensor: Generated noise tensor with specified configuration
        """
        # Set seed for reproducible noise generation
        if self.noise_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.noise_seed)
        else:
            generator = None
            
        if self.noise_type == 'gaussian':
            # Generate Gaussian noise with specified mean and std
            if generator is not None:
                noise = torch.normal(mean=self.noise_mean, std=self.noise_std, 
                                   size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.normal(mean=self.noise_mean, std=self.noise_std, 
                                   size=shape, device=device, dtype=dtype)
                                   
        elif self.noise_type == 'uniform':
            # Generate uniform noise in range [mean-std, mean+std]
            low = self.noise_mean - self.noise_std
            high = self.noise_mean + self.noise_std
            if generator is not None:
                noise = torch.rand(size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.rand(size=shape, device=device, dtype=dtype)
            noise = noise * (high - low) + low
            
        elif self.noise_type == 'zero':
            # Generate zero noise (no replacement effect)
            noise = torch.zeros(size=shape, device=device, dtype=dtype)
            
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}. "
                           f"Supported types: 'gaussian', 'uniform', 'zero'")
        
        return noise

    def _validate_station_specific_prob_matrix(self, prob_matrix, expected_shape):
        """
        Validate station-specific prob_matrix dimensions and values.
        
        Args:
            prob_matrix: Probability matrix to validate [N, 7, 56]
            expected_shape: Expected shape tuple (N, 7, 56)
            
        Returns:
            validated_prob_matrix: Validated and potentially corrected prob_matrix
        """
        if prob_matrix is None:
            raise ValueError(f" prob_matrix is None ") 
            
        # Check dimensions
        if len(prob_matrix.shape) != 3:
            raise ValueError(f"Station-specific prob_matrix must be 3D [N, 7, 56], got shape: {prob_matrix.shape}")
            
        if prob_matrix.shape != expected_shape:
            raise ValueError(f"prob_matrix shape {prob_matrix.shape} doesn't match "
                           f"expected shape {expected_shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(prob_matrix).any() or torch.isinf(prob_matrix).any():
            print("Warning: prob_matrix contains NaN or infinite values, clamping to [0, 1]")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        # Ensure values are in [0, 1] range
        if (prob_matrix < 0).any() or (prob_matrix > 1).any():
            print("Warning: prob_matrix values outside [0, 1] range, clamping")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        return prob_matrix

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores: Q @ K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Apply scaling and softmax to get attention weights A
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
    
        # Use attn_mask directly as prob_matrix tensor
        prob_matrix = attn_mask  # Expected shape: [N, 7, 56]
        
        # Handle station-specific prob_matrix [N, 7, 56]
        if len(prob_matrix.shape) == 3:
            N, aq_features, mete_features = prob_matrix.shape
            
            # Validate prob_matrix
            prob_matrix = self._validate_station_specific_prob_matrix(prob_matrix, (N, aq_features, mete_features))
            
            # For station-specific processing, we need to handle each station independently
            # Input shapes: queries [B*N, L, H, E], keys [B*N, S, H, E], values [B*N, S, H, D]
            # where B*N represents batch_size * num_stations
            
            if B % N != 0:
                raise ValueError(f"Batch size {B} is not divisible by number of stations {N}")
            
            actual_batch_size = B // N
            
            # Reshape to separate batch and station dimensions
            # [B*N, L, H, E] -> [actual_batch_size, N, L, H, E]
            values_reshaped = values.view(actual_batch_size, N, S, H, D)
            A_reshaped = A.view(actual_batch_size, N, H, L, S)
            
            # Expand prob_matrix to match attention dimensions
            # [N, 7, 56] -> [actual_batch_size, N, H, 7, 56]
            P = prob_matrix.unsqueeze(0).unsqueeze(2).expand(actual_batch_size, N, H, aq_features, mete_features)
            
            # Ensure P is on the same device as other tensors
            if P.device != values.device:
                P = P.to(values.device)
            
            # Generate noise tensor Z with same shape as values_reshaped
            Z = self.generate_noise(values_reshaped.shape, values_reshaped.device, values_reshaped.dtype)
            
            # Apply station-specific backdoor replacement using matrix operations
            # Vectorized computation for all stations simultaneously
            
            # Apply backdoor replacement: O = (A .* P) @ V + (A .* (1-P)) @ Z
            weighted_attention_v = A_reshaped * P  # [actual_batch_size, N, H, L, S]
            weighted_attention_z = A_reshaped * (1.0 - P)  # [actual_batch_size, N, H, L, S]
            
            # Matrix multiplication using einsum for vectorized computation
            # [actual_batch_size, N, H, L, S] @ [actual_batch_size, N, S, H, D] -> [actual_batch_size, N, L, H, D]
            V_original = torch.einsum("bnhls,bnshd->bnlhd", weighted_attention_v, values_reshaped)
            V_noise = torch.einsum("bnhls,bnshd->bnlhd", weighted_attention_z, Z)
            
            # Final output: combination of original and noise components
            V_combined = V_original + V_noise  # [actual_batch_size, N, L, H, D]
            
            # Reshape back to original format: [actual_batch_size, N, L, H, D] -> [B, L, H, D]
            V = V_combined.view(B, L, H, D)
                
        else:
            raise ValueError(f"FullAttention_v5 expects prob_matrix with shape [N, 7, 56], "
                            f"got shape: {prob_matrix.shape}")
            
 

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention_v42(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 noise_type='gaussian', noise_std=0.1, noise_mean=0.0, noise_seed=42):
        super(FullAttention_v42, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Noise configuration parameters
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise_seed = noise_seed
        
        # Set random seed for reproducibility
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)
 
    def generate_noise(self, shape, device, dtype, values):
        """
        Generate configurable noise tensor matching the shape and device of input tensor.
        Uses values tensor's mean and variance instead of fixed parameters.
        
        Args:
            shape: Target tensor shape
            device: Target device (cuda/cpu)
            dtype: Target data type
            values: Input values tensor to compute statistics from
            
        Returns:
            noise_tensor: Generated noise tensor with specified configuration
        """
        # Set seed for reproducible noise generation
        if self.noise_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.noise_seed)
        else:
            generator = None
        
        # Compute mean and std from values tensor
        values_mean = values.mean().item()
        values_std = values.std().item()
        
        # Prevent zero std which would cause issues
        if values_std < 1e-8:
            values_std = 1e-8
            
        if self.noise_type == 'gaussian':
            # Generate Gaussian noise with values' mean and std
            if generator is not None:
                noise = torch.normal(mean=values_mean, std=values_std, 
                                   size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.normal(mean=values_mean, std=values_std, 
                                   size=shape, device=device, dtype=dtype)
                                   
        elif self.noise_type == 'uniform':
            # Generate uniform noise in range [mean-std, mean+std]
            low = values_mean - values_std
            high = values_mean + values_std
            if generator is not None:
                noise = torch.rand(size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.rand(size=shape, device=device, dtype=dtype)
            noise = noise * (high - low) + low
            
        elif self.noise_type == 'zero':
            # Generate zero noise (no replacement effect)
            noise = torch.zeros(size=shape, device=device, dtype=dtype)
            
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}. "
                           f"Supported types: 'gaussian', 'uniform', 'zero'")
        
        return noise

    def _validate_prob_matrix(self, prob_matrix, expected_shape):
        """
        Validate prob_matrix dimensions and values.
        
        Args:
            prob_matrix: Probability matrix to validate
            expected_shape: Expected shape tuple (N, N)
            
        Returns:
            validated_prob_matrix: Validated and potentially corrected prob_matrix
        """
        if prob_matrix is None:
            raise ValueError(f" prob_matrix is None ") 
            
        # Check dimensions
        if len(prob_matrix.shape) != 2:
            raise ValueError(f"prob_matrix must be 2D, got shape: {prob_matrix.shape}")
            
        if prob_matrix.shape != expected_shape:
            raise ValueError(f"prob_matrix shape {prob_matrix.shape} doesn't match "
                           f"expected shape {expected_shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(prob_matrix).any() or torch.isinf(prob_matrix).any():
            print("Warning: prob_matrix contains NaN or infinite values, clamping to [0, 1]")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        # Ensure values are in [0, 1] range
        if (prob_matrix < 0).any() or (prob_matrix > 1).any():
            print("Warning: prob_matrix values outside [0, 1] range, clamping")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        return prob_matrix

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores: Q @ K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Apply scaling and softmax to get attention weights A
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
 
        # Use attn_mask directly as prob_matrix tensor
        prob_matrix = attn_mask  # Expected shape: [N, N] or [B, H, N, N]
        
        # Handle different prob_matrix shapes
        if len(prob_matrix.shape) == 2:
            # prob_matrix is [N, N], need to expand to [B, H, N, N]
            N1, N2 = prob_matrix.shape
            if N1 != L or N2 != S:
                raise ValueError(f"prob_matrix shape [{N1}, {N2}] doesn't match "
                                f"sequence lengths [L={L}, S={S}]")
            
            # Validate prob_matrix
            prob_matrix = self._validate_prob_matrix(prob_matrix, (N1, N2))
            
            # Expand to batch and head dimensions: [N, N] -> [B, H, N, N]
            P = prob_matrix.unsqueeze(0).unsqueeze(0).expand(B, H, L, S)
            
        else:
            raise ValueError(f"prob_matrix must be 2D [N, N] or 4D [B, H, N, N], "
                            f"got shape: {prob_matrix.shape}")
        
        # Ensure P is on the same device as other tensors
        if P.device != values.device:
            P = P.to(values.device)
        
        # Generate noise tensor Z with same shape as values, using values statistics
        Z = self.generate_noise(values.shape, values.device, values.dtype, values)
        
        # Apply backdoor replacement using matrix operations: O = (A .* P) @ V + (A .* (1-P)) @ Z
        # A .* P: element-wise multiplication of attention weights with prob_matrix
        weighted_attention_v = A * P  # [B, H, L, S]
        weighted_attention_z = A * (1.0 - P)  # [B, H, L, S]
        
        # Matrix multiplication: [B, H, L, S] @ [B, S, H, D] -> [B, L, H, D]
        V_original = torch.einsum("bhls,bshd->blhd", weighted_attention_v, values)
        V_noise = torch.einsum("bhls,bshd->blhd", weighted_attention_z, Z)
        
        # Final output: combination of original and noise components
        V = V_original + V_noise
            
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention_v52(nn.Module):
    """
    FullAttention_v52: Enhanced version supporting station-specific variable-level prob_matrix
    Supports prob_matrix with shape [N, 7, 56] where N is number of stations,
    7 is aq features, 56 is mete features
    Uses values tensor's mean and variance for noise generation instead of fixed parameters.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 noise_type='gaussian', noise_std=0.1, noise_mean=0.0, noise_seed=42):
        super(FullAttention_v52, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Noise configuration parameters
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise_seed = noise_seed
        
        # Set random seed for reproducibility
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)

    def generate_noise(self, shape, device, dtype, values):
        """
        Generate configurable noise tensor matching the shape and device of input tensor.
        Uses values tensor's mean and variance instead of fixed parameters.
        
        Args:
            shape: Target tensor shape
            device: Target device (cuda/cpu)
            dtype: Target data type
            values: Input values tensor to compute statistics from
            
        Returns:
            noise_tensor: Generated noise tensor with specified configuration
        """
        # Set seed for reproducible noise generation
        if self.noise_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.noise_seed)
        else:
            generator = None
        
        # Compute mean and std from values tensor
        values_mean = values.mean().item()
        values_std = values.std().item()
        
        # Prevent zero std which would cause issues
        if values_std < 1e-8:
            values_std = 1e-8
            
        if self.noise_type == 'gaussian':
            # Generate Gaussian noise with values' mean and std
            if generator is not None:
                noise = torch.normal(mean=values_mean, std=values_std, 
                                   size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.normal(mean=values_mean, std=values_std, 
                                   size=shape, device=device, dtype=dtype)
                                   
        elif self.noise_type == 'uniform':
            # Generate uniform noise in range [mean-std, mean+std]
            low = values_mean - values_std
            high = values_mean + values_std
            if generator is not None:
                noise = torch.rand(size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.rand(size=shape, device=device, dtype=dtype)
            noise = noise * (high - low) + low
            
        elif self.noise_type == 'zero':
            # Generate zero noise (no replacement effect)
            noise = torch.zeros(size=shape, device=device, dtype=dtype)
            
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}. "
                           f"Supported types: 'gaussian', 'uniform', 'zero'")
        
        return noise

    def _validate_station_specific_prob_matrix(self, prob_matrix, expected_shape):
        """
        Validate station-specific prob_matrix dimensions and values.
        
        Args:
            prob_matrix: Probability matrix to validate [N, 7, 56]
            expected_shape: Expected shape tuple (N, 7, 56)
            
        Returns:
            validated_prob_matrix: Validated and potentially corrected prob_matrix
        """
        if prob_matrix is None:
            raise ValueError(f" prob_matrix is None ") 
            
        # Check dimensions
        if len(prob_matrix.shape) != 3:
            raise ValueError(f"Station-specific prob_matrix must be 3D [N, 7, 56], got shape: {prob_matrix.shape}")
            
        if prob_matrix.shape != expected_shape:
            raise ValueError(f"prob_matrix shape {prob_matrix.shape} doesn't match "
                           f"expected shape {expected_shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(prob_matrix).any() or torch.isinf(prob_matrix).any():
            print("Warning: prob_matrix contains NaN or infinite values, clamping to [0, 1]")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        # Ensure values are in [0, 1] range
        if (prob_matrix < 0).any() or (prob_matrix > 1).any():
            print("Warning: prob_matrix values outside [0, 1] range, clamping")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        return prob_matrix

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores: Q @ K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Apply scaling and softmax to get attention weights A
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
    
        # Use attn_mask directly as prob_matrix tensor
        prob_matrix = attn_mask  # Expected shape: [N, 7, 56]
        
        # Handle station-specific prob_matrix [N, 7, 56]
        if len(prob_matrix.shape) == 3:
            N, aq_features, mete_features = prob_matrix.shape
            
            # Validate prob_matrix
            prob_matrix = self._validate_station_specific_prob_matrix(prob_matrix, (N, aq_features, mete_features))
            
            # For station-specific processing, we need to handle each station independently
            # Input shapes: queries [B*N, L, H, E], keys [B*N, S, H, E], values [B*N, S, H, D]
            # where B*N represents batch_size * num_stations
            
            if B % N != 0:
                raise ValueError(f"Batch size {B} is not divisible by number of stations {N}")
            
            actual_batch_size = B // N
            
            # Reshape to separate batch and station dimensions
            # [B*N, L, H, E] -> [actual_batch_size, N, L, H, E]
            values_reshaped = values.view(actual_batch_size, N, S, H, D)
            A_reshaped = A.view(actual_batch_size, N, H, L, S)
            
            # Expand prob_matrix to match attention dimensions
            # [N, 7, 56] -> [actual_batch_size, N, H, 7, 56]
            P = prob_matrix.unsqueeze(0).unsqueeze(2).expand(actual_batch_size, N, H, aq_features, mete_features)
            
            # Ensure P is on the same device as other tensors
            if P.device != values.device:
                P = P.to(values.device)
            
            # Generate noise tensor Z with same shape as values_reshaped, using values statistics
            Z = self.generate_noise(values_reshaped.shape, values_reshaped.device, values_reshaped.dtype, values_reshaped)
            
            # Apply station-specific backdoor replacement using matrix operations
            # Vectorized computation for all stations simultaneously
            
            # Apply backdoor replacement: O = (A .* P) @ V + (A .* (1-P)) @ Z
            weighted_attention_v = A_reshaped * P  # [actual_batch_size, N, H, L, S]
            weighted_attention_z = A_reshaped * (1.0 - P)  # [actual_batch_size, N, H, L, S]
            
            # Matrix multiplication using einsum for vectorized computation
            # [actual_batch_size, N, H, L, S] @ [actual_batch_size, N, S, H, D] -> [actual_batch_size, N, L, H, D]
            V_original = torch.einsum("bnhls,bnshd->bnlhd", weighted_attention_v, values_reshaped)
            V_noise = torch.einsum("bnhls,bnshd->bnlhd", weighted_attention_z, Z)
            
            # Final output: combination of original and noise components
            V_combined = V_original + V_noise  # [actual_batch_size, N, L, H, D]
            
            # Reshape back to original format: [actual_batch_size, N, L, H, D] -> [B, L, H, D]
            V = V_combined.view(B, L, H, D)
                
        else:
            raise ValueError(f"FullAttention_v52 expects prob_matrix with shape [N, 7, 56], "
                            f"got shape: {prob_matrix.shape}")
            
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention_v6(nn.Module):
    """
    FullAttention_v6: Variable-level attention with prob_matrix for AQ-Mete causal relationships
    Supports prob_matrix with shape [7, 56] where 7 is aq features, 56 is mete features
    Used for meteorological updates in v9.191
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 noise_type='gaussian', noise_std=0.1, noise_mean=0.0, noise_seed=42):
        super(FullAttention_v6, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Noise configuration parameters
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise_seed = noise_seed
        
        # Set random seed for reproducibility
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)

    def generate_noise(self, shape, device, dtype):
        """
        Generate configurable noise tensor matching the shape and device of input tensor.
        
        Args:
            shape: Target tensor shape
            device: Target device (cuda/cpu)
            dtype: Target data type
            
        Returns:
            noise_tensor: Generated noise tensor with specified configuration
        """
        # Set seed for reproducible noise generation
        if self.noise_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.noise_seed)
        else:
            generator = None
            
        if self.noise_type == 'gaussian':
            # Generate Gaussian noise with specified mean and std
            if generator is not None:
                noise = torch.normal(mean=self.noise_mean, std=self.noise_std, 
                                   size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.normal(mean=self.noise_mean, std=self.noise_std, 
                                   size=shape, device=device, dtype=dtype)
                                   
        elif self.noise_type == 'uniform':
            # Generate uniform noise in range [mean-std, mean+std]
            low = self.noise_mean - self.noise_std
            high = self.noise_mean + self.noise_std
            if generator is not None:
                noise = torch.rand(size=shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.rand(size=shape, device=device, dtype=dtype)
            noise = noise * (high - low) + low
            
        elif self.noise_type == 'zero':
            # Generate zero noise (no replacement effect)
            noise = torch.zeros(size=shape, device=device, dtype=dtype)
            
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}. "
                           f"Supported types: 'gaussian', 'uniform', 'zero'")
        
        return noise

    def _validate_variable_prob_matrix(self, prob_matrix, expected_shape):
        """
        Validate variable-level prob_matrix dimensions and values.
        
        Args:
            prob_matrix: Probability matrix to validate [7, 56]
            expected_shape: Expected shape tuple (7, 56)
            
        Returns:
            validated_prob_matrix: Validated and potentially corrected prob_matrix
        """
        if prob_matrix is None:
            raise ValueError(f" prob_matrix is None ") 
            
        # Check dimensions
        if len(prob_matrix.shape) != 2:
            raise ValueError(f"Variable-level prob_matrix must be 2D [7, 56], got shape: {prob_matrix.shape}")
            
        if prob_matrix.shape != expected_shape:
            raise ValueError(f"prob_matrix shape {prob_matrix.shape} doesn't match "
                           f"expected shape {expected_shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(prob_matrix).any() or torch.isinf(prob_matrix).any():
            print("Warning: prob_matrix contains NaN or infinite values, clamping to [0, 1]")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        # Ensure values are in [0, 1] range
        if (prob_matrix < 0).any() or (prob_matrix > 1).any():
            print("Warning: prob_matrix values outside [0, 1] range, clamping")
            prob_matrix = torch.clamp(prob_matrix, 0.0, 1.0)
            
        return prob_matrix

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores: Q @ K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Apply scaling and softmax to get attention weights A
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # Check if variable-level backdoor replacement should be applied
        if (self.mask_flag and attn_mask is not None and 
            hasattr(attn_mask, 'prob_matrix') and attn_mask.prob_matrix is not None):
            
            # Get prob_matrix from attn_mask
            prob_matrix = attn_mask.prob_matrix  # Expected shape: [7, 56]
            
            # Handle variable-level prob_matrix [7, 56]
            if len(prob_matrix.shape) == 2:
                aq_features, mete_features = prob_matrix.shape
                
                # For meteorological updates: queries are AQ features [B*N, 7, H, E], 
                # keys/values are mete features [B*N, 56, H, E/D]
                if L != aq_features or S != mete_features:
                    print(f"DEBUG: queries shape: {queries.shape}, keys shape: {keys.shape}, values shape: {values.shape}")
                    print(f"DEBUG: B={B}, L={L}, H={H}, E={E}, S={S}")
                    print(f"DEBUG: prob_matrix shape: {prob_matrix.shape}")
                    print(f"DEBUG: Expected aq_features={aq_features}, mete_features={mete_features}")
                    raise ValueError(f"prob_matrix shape [{aq_features}, {mete_features}] doesn't match "
                                   f"sequence lengths [L={L}, S={S}]. Expected L={aq_features}, S={mete_features}")
                
                # Validate prob_matrix
                prob_matrix = self._validate_variable_prob_matrix(prob_matrix, (aq_features, mete_features))
                
                # Expand to batch and head dimensions: [7, 56] -> [B, H, 7, 56]
                P = prob_matrix.unsqueeze(0).unsqueeze(0).expand(B, H, L, S)
                
                # Ensure P is on the same device as other tensors
                if P.device != values.device:
                    P = P.to(values.device)
                
                # Generate noise tensor Z with same shape as values
                Z = self.generate_noise(values.shape, values.device, values.dtype)
                
                # Apply backdoor replacement: O = (A .* P) @ V + (A .* (1-P)) @ Z
                # A .* P: element-wise multiplication of attention weights with prob_matrix
                weighted_attention_v = A * P  # [B, H, L, S]
                weighted_attention_z = A * (1.0 - P)  # [B, H, L, S]
                
                # Matrix multiplication: [B, H, L, S] @ [B, S, H, D] -> [B, L, H, D]
                V_original = torch.einsum("bhls,bshd->blhd", weighted_attention_v, values)
                V_noise = torch.einsum("bhls,bshd->blhd", weighted_attention_z, Z)
                
                # Final output: combination of original and noise components
                V = V_original + V_noise
                
            else:
                raise ValueError(f"FullAttention_v6 expects prob_matrix with shape [7, 56], "
                               f"got shape: {prob_matrix.shape}")
            
        else:
            # Standard attention without backdoor replacement
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None
