from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import torch
from utils.dist import get_world_size
from scipy.spatial.distance import cdist
import pandas
from torch.utils.data import DataLoader,Dataset, RandomSampler
from sklearn.preprocessing import StandardScaler

# from torch_geometric.data import InMemoryDataset, Data , Batch
# import torch_geometric
# from torch_geometric.data import DataLoader
from torch.utils.data.dataloader import default_collate
def gat_lstmcollate_fn(data):
    aq_train_data = []
    mete_train_data = []
    # aq_g_list = []
    # mete_g_list = []
    # edge_index = []
    # edge_attr = []
    # pos = []
    label = []
    reconstructed_label = []
    # dec_input = []
    for unit in data:
        aq_train_data.append(unit[0]['aq_train_data'])
        mete_train_data.append(unit[0]['mete_train_data'])
   
        # aq_g_list = aq_g_list + unit[0]['aq_g_list']
        # mete_g_list = mete_g_list + unit[0]['mete_g_list']
        # edge_index.append(unit[0]['edge_index'])
        # edge_attr.append(unit[0]['edge_attr'])
        # pos.append(unit[0]['pos'])
        # dec_input.append(unit[0]['dec_input'])
        label.append(unit[1])
        reconstructed_label.append(unit[2])

    AQStation_coordinate = data[0][0]['AQStation_coordinate']
    label = torch.stack(label)
    label = torch.transpose(label, 1, 2)
    label = torch.flatten(label, start_dim=0, end_dim=1)
    reconstructed_label = torch.stack(reconstructed_label)
    reconstructed_label = torch.transpose(reconstructed_label, 1, 2)
    reconstructed_label = torch.flatten(reconstructed_label, start_dim=0, end_dim=1)



    return {
        'aq_train_data': torch.stack(aq_train_data),
        'AQStation_coordinate':AQStation_coordinate,
        'mete_train_data': torch.stack(mete_train_data),
    #     'aq_G': Batch.from_data_list(aq_g_list),
    #    'mete_G': Batch.from_data_list(mete_g_list),
       # 'edge_index': torch.stack(edge_index).int(),
       # 'edge_attr': torch.stack(edge_attr),
       # 'pos': torch.stack(pos),
        # 'dec_input': torch.stack(dec_input),
            } , {
        'label': label,
        'reconstructed_label':reconstructed_label,
            }

class stsdataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self,args, data_dir, batch_size, shuffle=True, num_workers=1, training=True, T=24, t=12, time_interval=1, collate_fn=gat_lstmcollate_fn):
        # import pdb
        # pdb.set_trace()
        self.T = T
        self.t = t

        self.dataset = STSDataset(args=args, data_path=data_dir,training=training, time_interval=time_interval)
        if get_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=shuffle)
        else:
            sampler = None
        super().__init__(self.dataset, batch_size, shuffle, num_workers, sampler=sampler,collate_fn=collate_fn)





class STSDataset(Dataset):

    def __init__(self,args,data_path,training,scale=True,time_interval=1):
        """
        root: 数据集保存的地方。
        会产生两个文件夹：
          raw_dir(downloaded dataset) 和 processed_dir(processed data)。
        """
#['year', 'month', 'day','week','weekday', 'hour', 'AQI','PM2.5','PM10', 'SO2', 'NO2', 'O3', 'CO']
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.scale = scale
        self.training = training
        self.time_interval = time_interval
        self.mete_scaler = StandardScaler()
        self.aq_scaler = StandardScaler()

        super().__init__()
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pandas.read_pickle(f)
        if 'metedata' in self.data:
            if self.scale and self.training:
                L, N, C = self.data['metedata'].shape
                self.metedata = self.data['metedata'].reshape(L * N, C)
                self.metedata = self.mete_scaler.fit_transform(self.metedata)
                self.metedata = self.metedata.reshape(L, N, C)
            else:
                self.metedata = self.data['metedata']
        else:
            self.metedata = None

        if 'AQdata' in self.data:
            self.AQStation_imformation = self.data['AQStation_imformation']
            self.AQStation_coordinate = self.get_coordinate()
            if self.scale and self.training:
                L, N, C = self.data['AQdata'].shape
                self.AQdata = self.data['AQdata'].reshape(L * N, C)
                self.AQdata[:, -7:] = self.aq_scaler.fit_transform(self.AQdata[:, -7:])
                self.AQdata = self.AQdata.reshape(L, N, C)
            else:
                self.AQdata = self.data['AQdata']
        
        # 为每个样本预先生成随机偏移量，确保分布均匀
        if self.time_interval > 1:
            dataset_length = (len(self.AQdata) - self.seq_len - self.pred_len) // self.time_interval
            # 为每个样本随机分配偏移量，确保均匀分布
            self.offsets = np.random.choice(self.time_interval, size=dataset_length)
        
        # # self.train_AQ = self.data['data']
        # self.aq_edge_index, self.aq_edge_attr, self.aq_node_coords = self.get_edge_attr(np.array(self.AQStation_imformation.iloc[:, -2:]).astype('float64'))
        # self.mete_edge_index, self.mete_edge_attr, self.mete_node_coords=self.get_edge_attr(np.array(self.meteStation_imformation.loc[:, ['经度', '纬度']]).astype('float64'))




    def __len__(self):
        # assert len(self.train_AQ) == len(self.labels)
        if self.time_interval > 1:
            # 使用更少的数据进行训练，数据量减少为原来的1/time_interval
            return (len(self.AQdata) - self.seq_len - self.pred_len) // self.time_interval
        else:
            return len(self.AQdata) - self.seq_len - self.pred_len


    def __getitem__(self, idx):
        input_data = {}
        
        # 根据时间间隔调整实际的起始索引
        if self.time_interval > 1:
            # 使用预生成的随机偏移量，确保数据分布均匀
            offset = self.offsets[idx]
            actual_idx = offset + idx * self.time_interval
        else:
            actual_idx = idx
            
        aq_train_data = torch.tensor(self.AQdata[actual_idx:actual_idx+self.seq_len+self.pred_len]).float()
        # if self.metedata != None:
        mete_train_data = torch.tensor(self.metedata[actual_idx:actual_idx + self.seq_len + self.pred_len]).float()
        # fore_train_data = torch.tensor(self.metedata[idx:idx + self.seq_len + self.pred_len]).float()
        # else: mete_train_data = None

        # if getattr(self, "AQStation_coordinate", None):
        #     AQStation_coordinate = 
        # else: AQStation_coordinate = None
        
        # train_data = torch.cat((train_data[:,:8,:], train_data[:,:9,:],), dim=1)
        # aq_g_list = [torch_geometric.data.Data(x=s, edge_index=self.aq_edge_index, edge_attr=self.aq_edge_attr.float(),
        #                                        pos=self.aq_node_coords.float()) for s in aq_train_data[:self.seq_len]]
        # mete_g_list = [torch_geometric.data.Data(x=s, edge_index=self.mete_edge_index, edge_attr=self.mete_edge_attr.float(),
        #                                        pos=self.mete_node_coords.float()) for s in mete_train_data[:self.seq_len]]

        # dec_input = aq_train_data[-(self.pred_len + 1):, :, :].clone().detach().float()


        label = aq_train_data[-self.pred_len:, :, -7:]
        reconstructed_label = torch.concat([aq_train_data[:self.seq_len, :, -7:],mete_train_data[:self.seq_len, :, :]],-1)



        # getattr(some_module, "some_attribute", None)
        data = {
            'aq_train_data': aq_train_data,
            'mete_train_data': mete_train_data,
            'AQStation_coordinate': self.AQStation_coordinate
            # 'aq_g_list': aq_g_list,
            # 'mete_g_list': mete_g_list,
            # 'edge_index': self.edge_index,
            # 'edge_attr': self.edge_attr.float(),
            # 'pos': self.node_coords.float(),
        }

        return data, label,reconstructed_label

    def get_coordinate(self, ):
        
        # self.AQStation_imformation.loc[1558, ['经度', '纬度']]=[ 112.333672,16.832701,]
        # self.AQStation_imformation.loc[201, ['经度', '纬度']]=[ 119.4200584,32.1901111,]
        # self.AQStation_imformation.loc[253, ['经度', '纬度']]=[ 120.7510971,30.7474425,]
        # self.AQStation_imformation.loc[345, ['经度', '纬度']]=[ 113.330215,23.135742,]
        # self.AQStation_imformation.loc[976, ['经度', '纬度']]=[ 111.15081,37.51761]
        # self.AQStation_imformation.loc[1373, ['经度', '纬度']]=[ 107.952461,26.566523,]
        
        # 兼容 pandas.DataFrame 和 numpy.ndarray
        if hasattr(self.AQStation_imformation, 'loc'):
            AQStation_imformation = np.array(self.AQStation_imformation.loc[:, ['经度', '纬度']]).astype('float64')
        else:
            AQStation_imformation = np.array(self.AQStation_imformation).astype('float64')

        # means = AQStation_imformation.mean(0,)
        # AQStation_imformation = AQStation_imformation - means
        # AQStation_imformation=AQStation_imformation/np.sqrt(AQStation_imformation.var(axis=0))
        return torch.tensor(AQStation_imformation).float()