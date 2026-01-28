from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import torch
from utils.dist import get_world_size
from scipy.spatial.distance import cdist
import pandas
from torch.utils.data import DataLoader, RandomSampler
from torch_geometric.data import InMemoryDataset, Data ,Dataset, Batch
import torch_geometric
from torch_geometric.data import DataLoader
from torch.utils.data.dataloader import default_collate
def gat_lstmcollate_fn(data):
    aq_train_data = []
    mete_train_data = []
    aq_g_list = []
    mete_g_list = []
    edge_index = []
    edge_attr = []
    pos = []
    label = []
    # dec_input = []
    for unit in data:
        aq_train_data.append(unit[0]['aq_train_data'])
        # mete_train_data.append(unit[0]['mete_train_data'])
        aq_g_list = aq_g_list + unit[0]['aq_g_list']
        # mete_g_list = mete_g_list + unit[0]['mete_g_list']
        # edge_index.append(unit[0]['edge_index'])
        # edge_attr.append(unit[0]['edge_attr'])
        # pos.append(unit[0]['pos'])
        # dec_input.append(unit[0]['dec_input'])
        label.append(unit[1])


    label = torch.stack(label)
    label = torch.transpose(label, 1, 2)
    label = torch.flatten(label, start_dim=0, end_dim=1)
    return {
        'aq_train_data': torch.stack(aq_train_data),
        # 'mete_train_data': torch.stack(mete_train_data),
        'aq_G': Batch.from_data_list(aq_g_list),
    #    'mete_G': Batch.from_data_list(mete_g_list),
       # 'edge_index': torch.stack(edge_index).int(),
       # 'edge_attr': torch.stack(edge_attr),
       # 'pos': torch.stack(pos),
        # 'dec_input': torch.stack(dec_input),
            } , label

class pygmmdataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self,args, data_dir, batch_size, shuffle=True, num_workers=1, training=True, T=24, t=12, collate_fn=gat_lstmcollate_fn):
        # import pdb
        # pdb.set_trace()
        self.T = T
        self.t = t

        self.dataset = AQGDataset(args=args, data_path=data_dir)
        if get_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=shuffle)
        else:
            sampler = None
        super().__init__(self.dataset, batch_size, shuffle, num_workers, sampler=sampler,collate_fn=collate_fn)





class AQGDataset(Dataset):

    def __init__(self,args,data_path, root='/PROTEINS_full', filepath='/PROTEINS_full/raw', name='custom',
                 use_edge_attr=True, transform=None,
                 pre_transform=None, pre_filter=None):
        """
        root: 数据集保存的地方。
        会产生两个文件夹：
          raw_dir(downloaded dataset) 和 processed_dir(processed data)。
        """

        self.name = name
        self.root = root
        self.filepath = filepath
        self.use_edge_attr = use_edge_attr
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        # super().__init__(root, transform, pre_transform, pre_filter)
        super().__init__()
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pandas.read_pickle(f)

        self.metedata = self.data['metedata']
        self.AQdata = self.data['AQdata']

        self.AQStation_imformation = self.data['AQStation_imformation']
        # self.meteStation_imformation = self.data['meteStation_imformation']
        # self.train_AQ = self.data['data']
        # mete_coords = np.array(self.meteStation_imformation.loc[:, ['经度', '纬度']]).astype('float64')
        # self.AQStation_imformation = self.data['AQStation_imformation']
        self.AQStation_imformation = self.data['AQStation_imformation']
        # self.AQStation_coordinate = self.get_coordinate()
        AQ_coords = self.get_coordinate()
        self.aq_edge_index, self.aq_edge_attr, self.aq_node_coords = self.get_edge_attr(AQ_coords)
        # self.mete_edge_index, self.mete_edge_attr, self.mete_node_coords=self.get_edge_attr(np.array(self.meteStation_imformation.loc[:, ['经度', '纬度']]).astype('float64'))

        # self.lut = self.find_nearest_point(AQ_coords, mete_coords)
        self.AQdata = np.concatenate((self.AQdata[:, :, -7:], self.metedata[:, :, -9:]), axis=2)


    @property
    def raw_dir(self):
        """原始文件的文件夹"""
        return self.filepath

    @property
    def processed_dir(self):
        """处理后文件的文件夹"""
        return self.filepath

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2',]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ]


    # def download(self):
    #     """这里不需要下载"""
    #     pass

    def len(self):
        # assert len(self.train_AQ) == len(self.labels)
        return len(self.AQdata)-self.seq_len-self.pred_len


    def get(self, idx):
        input_data = {}
        aq_train_data = torch.tensor(self.AQdata[idx:idx+self.seq_len+self.pred_len]).float()
        # mete_train_data = torch.tensor(self.metedata[idx:idx + self.seq_len + self.pred_len]).float()

        # train_data = torch.cat((train_data[:,:8,:], train_data[:,:9,:],), dim=1)
        aq_g_list = [torch_geometric.data.Data(x=s, edge_index=self.aq_edge_index, edge_attr=self.aq_edge_attr.float(),
                                               pos=self.aq_node_coords.float()) for s in aq_train_data[:self.seq_len]]
        # mete_g_list = [torch_geometric.data.Data(x=s, edge_index=self.mete_edge_index, edge_attr=self.mete_edge_attr.float(),
                                            #    pos=self.mete_node_coords.float()) for s in mete_train_data[:self.seq_len]]


        # dec_input = aq_train_data[-(self.pred_len + 1):, :, :].clone().detach().float()






        label = aq_train_data[-self.pred_len:, :, 6:13]

        data = {
            'aq_train_data': aq_train_data,
            # 'mete_train_data': mete_train_data,
            'aq_g_list': aq_g_list,
            # 'mete_g_list': mete_g_list,
            # 'edge_index': self.edge_index,
            # 'edge_attr': self.edge_attr.float(),
            # 'pos': self.node_coords.float(),
        }

        return data, label

    def get_edge_attr(self, node_coords,    threshold=0.2):
        node_coords = torch.tensor(node_coords)
        # self.node_coords = torch.cat((self.node_coords[:8], self.node_coords[9:]), dim=0)
        # 计算节点之间的距离矩阵
        dist_matrix = cdist(node_coords, node_coords)

        # 设置阈值


        # 根据距离矩阵和阈值确定节点之间的连接关系
        edge_index = np.where(dist_matrix < threshold)

        # 转换为PyTorch Geometric所需的格式
        edge_index = torch.LongTensor(edge_index)

        # 获取边的起始节点和目标节点
        start_nodes, end_nodes = edge_index

        # 计算每条边的长度
        edge_lengths = dist_matrix[start_nodes, end_nodes]

        # 计算每条边的方向
        edge_directions = node_coords[end_nodes] - node_coords[start_nodes]
        edge_attr = torch.tensor(np.concatenate((edge_lengths[:, np.newaxis], edge_directions), axis=1))

        return edge_index, edge_attr, node_coords

    def find_nearest_point(self,A, B):
        nearest_indices = []
        for a in A:
            distances = [np.linalg.norm(a - b) for b in B]
            nearest_indices.append(np.argmin(distances))
        return nearest_indices

    def get_coordinate(self, ):
        
        # self.AQStation_imformation.loc[1558, ['经度', '纬度']]=[ 112.333672,16.832701,]
        # self.AQStation_imformation.loc[201, ['经度', '纬度']]=[ 119.4200584,32.1901111,]
        # self.AQStation_imformation.loc[253, ['经度', '纬度']]=[ 120.7510971,30.7474425,]
        # self.AQStation_imformation.loc[345, ['经度', '纬度']]=[ 113.330215,23.135742,]
        # self.AQStation_imformation.loc[976, ['经度', '纬度']]=[ 111.15081,37.51761]
        # self.AQStation_imformation.loc[1373, ['经度', '纬度']]=[ 107.952461,26.566523,]
        AQStation_imformation = np.array(self.AQStation_imformation.loc[:, ['经度', '纬度']]).astype('float64')

        # means = AQStation_imformation.mean(0,)
        # AQStation_imformation = AQStation_imformation - means
        # AQStation_imformation=AQStation_imformation/np.sqrt(AQStation_imformation.var(axis=0))
        return torch.tensor(AQStation_imformation).float()