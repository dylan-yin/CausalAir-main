import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import math

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weeknum_size = 53
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.weeknum_embed = Embed(weeknum_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.Temporal_feature = ['month', 'day', 'week', 'weekday', 'hour']

    def forward(self, x):
        x = x.long()
        for idx, freq in enumerate(self.Temporal_feature):
            if freq == 'year':
                pass
            elif freq == 'month':
                month_x = self.month_embed(x[:, :, idx])
            elif freq == 'day':
                day_x = self.day_embed(x[:, :, idx])
            elif freq == 'week':
                weeknum_x = self.weeknum_embed(x[:, :, idx])
            elif freq == 'weekday':
                weekday_x = self.weekday_embed(x[:, :, idx])
            elif freq == 'hour':
                hour_x = self.hour_embed(x[:, :, idx])

        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        # hour_x = self.hour_embed(x[:, :, 3])
        # weekday_x = self.weekday_embed(x[:, :, 2])
        # day_x = self.day_embed(x[:, :, 1])
        # month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + weeknum_x + day_x + month_x #+ minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class ChannelEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ChannelEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(c_in, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, ):
        return self.pe
    
class SpatialEmbedding(nn.Module):
    def __init__(self, coordinate, d_model):
        super(SpatialEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # means = AQStation_imformation.mean(0,)
        # AQStation_imformation = AQStation_imformation - means
        # AQStation_imformation=AQStation_imformation/np.sqrt(AQStation_imformation.var(axis=0))
        means = coordinate.mean(0,)
        coordinate = coordinate - means

        pe = torch.zeros(coordinate.size()[0], d_model).float()
        pe.require_grad = False

        # position = torch.arange(0, c_in).float().unsqueeze(1)
        position_x = coordinate[:,0].float().unsqueeze(1)
        position_y = coordinate[:,1].float().unsqueeze(1)

        div_term = (torch.arange(0, d_model, 4).float() * -(math.log(10000.0) / d_model)).exp().to(position_x.device)

        pe[:, 0::4] = torch.sin(position_x * div_term)
        pe[:, 1::4] = torch.cos(position_x* div_term)
        pe[:, 2::4] = torch.sin(position_y * div_term)
        pe[:, 3::4] = torch.cos(position_y * div_term)
        

        pe = pe.unsqueeze(0).to(position_x.device)
        self.tensor = pe
        self.register_buffer('pe', self.tensor)

        # self.pe = pe
        # self.register_buffer('pe', pe)

    def forward(self, ):
        return self.pe

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)

class DataEmbedding_st(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, d_model, aq_c_in, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_st, self).__init__()
        self.d_model = d_model
        self.aq_c_in = aq_c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.value_embedding = nn.Sequential(nn.Linear(seq_len, d_model),
                                             nn.GELU(),
                                             nn.Dropout(p=dropout),
                                             nn.Linear(d_model, d_model, bias=True),
                                       )
        self.fore_embedding = nn.Sequential(nn.Linear(pred_len, d_model),
                                             nn.GELU(),
                                             nn.Dropout(p=dropout),
                                             nn.Linear(d_model, d_model, bias=True),
                                       )
        self.spatial_embedding = None
        self.channel_embedding = ChannelEmbedding(c_in=enc_in,d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        # 可学习的mask token，用于替换被遮盖通道的嵌入
        future_token = nn.Parameter(torch.zeros(d_model),requires_grad=True) 
        self.register_parameter("future_token", future_token)
        nn.init.kaiming_normal_(future_token.unsqueeze(0))

    def forward(self, x, fore_x=None, coordinate=None):
        B, N, C, L = x.shape
        x = x.permute(0, 1, 3, 2)
        
        # x: [Batch Variate Time]
        if coordinate is None:
            x = self.value_embedding(x)
        else:
            if self.spatial_embedding is None:
                self.spatial_embedding = SpatialEmbedding(coordinate, self.d_model)
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(x) + self.channel_embedding() + self.spatial_embedding().unsqueeze(2)
            
            # Create future_aq_x from learnable tokens and embeddings
            future_aq_x = self.future_token.view(1, 1, 1, self.d_model) + \
                          self.spatial_embedding().unsqueeze(2) + \
                          self.channel_embedding()[:, :self.aq_c_in, :].unsqueeze(1)
            future_aq_x = future_aq_x.expand(B, -1, -1, -1)

            if fore_x is not None:
                fore_x = fore_x.permute(0, 1, 3, 2)
                fore_x = self.fore_embedding(fore_x) + self.channel_embedding()[:,7:,:] + self.spatial_embedding().unsqueeze(2)
                return self.dropout(x), self.dropout(fore_x), self.dropout(future_aq_x), self.spatial_embedding()
            else:
                return self.dropout(x), fore_x, self.dropout(future_aq_x), self.spatial_embedding()

class DataEmbedding_st_v2(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, d_model, aq_features, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_st_v2, self).__init__()
        self.d_model = d_model
        self.aq_features = aq_features
        
        # 历史值嵌入（包括aq和mete特征）
        self.hist_value_embedding = nn.Sequential(nn.Linear(seq_len, d_model),
                                             nn.GELU(),
                                             nn.Dropout(p=dropout),
                                             nn.Linear(d_model, d_model, bias=True))
        
        # 未来值嵌入（用于映射预测的未来aq值）
        self.future_value_embedding = nn.Sequential(nn.Linear(pred_len, d_model),
                                             nn.GELU(),
                                             nn.Dropout(p=dropout),
                                             nn.Linear(d_model, d_model, bias=True))
                                       
        self.spatial_embedding = None
        self.channel_embedding = ChannelEmbedding(c_in=enc_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, aq_data, mete_data, coordinate, is_future=False):
        if self.spatial_embedding is None:
            self.spatial_embedding = SpatialEmbedding(coordinate, self.d_model).to(aq_data.device)
        
        if is_future:
            # 处理未来AQ和气象数据
            aq_x = aq_data.permute(0, 1, 3, 2)
            aq_emb = self.future_value_embedding(aq_x)
            aq_embedding = aq_emb + self.channel_embedding()[:, :self.aq_features, :] + self.spatial_embedding().unsqueeze(2)
            
            mete_x = mete_data.permute(0, 1, 3, 2)
            mete_emb = self.future_value_embedding(mete_x)
            mete_embedding = mete_emb + self.channel_embedding()[:, self.aq_features:, :] + self.spatial_embedding().unsqueeze(2)
            
            return self.dropout(aq_embedding), self.dropout(mete_embedding)
        else:
            # 处理历史AQ和气象数据
            aq_x = aq_data.permute(0, 1, 3, 2)
            aq_emb = self.hist_value_embedding(aq_x)
            aq_embedding = aq_emb + self.channel_embedding()[:, :self.aq_features, :] + self.spatial_embedding().unsqueeze(2)
            
            mete_x = mete_data.permute(0, 1, 3, 2)
            mete_emb = self.hist_value_embedding(mete_x)
            mete_embedding = mete_emb + self.channel_embedding()[:, self.aq_features:, :] + self.spatial_embedding().unsqueeze(2)
            
            return self.dropout(aq_embedding), self.dropout(mete_embedding)


class Timestamp_Embedding(nn.Module):   
    def __init__(self, time_c, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(Timestamp_Embedding, self).__init__()
        self.d_model = d_model
        self.time_c = time_c
        
        self.time_embedding_List = nn.ModuleList([ nn.Linear(seq_len, d_model) for i in range(time_c)])
        self.fore_embedding = nn.Linear(48, d_model)
        self.spatial_embedding = None
        self.channel_embedding = ChannelEmbedding(c_in=time_c,d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, time_stamp):
        Embeded_timefearture=[]
        for idx, lays in enumerate(self.time_embedding_List ):
            Embeded_timefearture.append( lays(time_stamp[:,idx,:]))
        return torch.stack(Embeded_timefearture,1)



class DataEmbedding_MSinverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)

class MultiScale_TSMLP(nn.Module):
    def __init__(self,  seq_len,enc_in, d_model, kernel_list, dropout=0.1):
        super(MultiScale_TSMLP, self).__init__()
        self.kernel_list = kernel_list
        self.kernel = [nn.Linear(k, d_model*k/seq_len) for k in self.kernel_list]

        self.spatial_embedding = None
        self.channel_embedding = ChannelEmbedding(c_in=enc_in,d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            for i, k in  enumerate(self.kernel_list):
                
                x = self.kernel[i](x)


        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)
        
