import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from os import listdir
import numpy as np
import os.path as osp
import scipy.io as sio
import numpy as np
import scipy.sparse as sp


class ADNIDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.num_tasks = 1
        self.task_type = 'classification'
        self.eval_metric = 'accuracy'

        super(ADNIDataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        path1 = osp.join(self.raw_dir, 'MDD_ADJ')
        path1_nf = osp.join(self.raw_dir, 'MDD_NF')
        path1_dw = osp.join(self.raw_dir, 'MDD_DW')

        files = os.listdir(path1)
        files_nf_ASD = os.listdir(path1_nf)
        files_dw = os.listdir(path1_dw)

        data_list_ASD = []
        for i in range(len(files)):

            nf = sio.loadmat(osp.join(path1_nf, files_nf_ASD[i]))
            x = nf['norm_matrix']
            x = np.nan_to_num(x)
            x = torch.Tensor(x)

            adj = sio.loadmat(osp.join(path1, files[i]))
            edge_index = adj['cropped_matrix']

            edge_index = np.nan_to_num(edge_index)
            edge_index_temp = sp.coo_matrix(edge_index)
            edge_weight = torch.Tensor(edge_index_temp.data)

            edge_index = torch.Tensor(edge_index)
            edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
            num_nodes = int(edge_index.max()) + 1

            dw = sio.loadmat(osp.join(path1_dw, files_dw[i]))
            dw_array=dw['correlation_matrices']
            dyn_weight_np = [dw_array[j, 0] for j in range(dw_array.shape[0])]
            dyn_weight = [torch.tensor(matrix) for matrix in dyn_weight_np]
            dyn_weight = torch.stack(dyn_weight).float()

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                        y=0)
            data.dyn_weight = dyn_weight
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list_ASD.append(data)

        path2 = osp.join(self.raw_dir, 'NC_ADJ')
        path2_nf = osp.join(self.raw_dir, 'NC_NF')
        path2_dw = osp.join(self.raw_dir, 'NC_DW')

        files = os.listdir(path2)
        files_nf_TC = os.listdir(path2_nf)
        files_dw = os.listdir(path2_dw)

        data_list_TC = []
        for i in range(len(files)):
            nf = sio.loadmat(osp.join(path2_nf, files_nf_TC[i]))
            x = nf['norm_matrix']
            x = np.nan_to_num(x)
            x = torch.Tensor(x)
            adj = sio.loadmat(osp.join(path2, files[i]))
            edge_index = adj['cropped_matrix']
            edge_index = np.nan_to_num(edge_index)
            edge_index_temp = sp.coo_matrix(edge_index)
            edge_weight = torch.Tensor(edge_index_temp.data)
            edge_index = torch.Tensor(edge_index)
            edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
            num_nodes = int(edge_index.max()) + 1

            dw = sio.loadmat(osp.join(path2_dw, files_dw[i]))
            dw_array = dw['correlation_matrices']
            dyn_weight_np = [dw_array[j, 0] for j in range(dw_array.shape[0])]
            dyn_weight = [torch.tensor(matrix) for matrix in dyn_weight_np]
            dyn_weight=torch.stack(dyn_weight).float()

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                        y=1)
            data.dyn_weight = dyn_weight

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list_TC.append(data)

        data_list = data_list_ASD + data_list_TC
        torch.save(self.collate(data_list),
                   osp.join(self.processed_dir, 'data.pt'))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
