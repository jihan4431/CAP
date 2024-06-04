import random
import numpy as np

import torch
from torch.utils.data import Dataset


op_list={'input':0,
         'nor_conv_1x1':1,
         'nor_conv_3x3':2,
         'avg_pool_3x3':3,
         'skip_connect':4,
         'none':5,
         'output':6}

def nasbench201_to_nasbench101(arch_list):
    num_ops = sum(range(1, 1 + len(arch_list))) + 2
    adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
    ops = ['input', 'output']
    node_lists = [[0]]
    for node_201 in arch_list:
        node_list = []
        for node in node_201:
            node_idx = len(ops) - 1
            adj[node_lists[node[1]], node_idx] = 1
            ops.insert(-1, node[0])
            node_list.append(node_idx)
        node_lists.append(node_list)
    adj[-(1+len(arch_list)):-1, -1] = 1
    arch = {'adj': adj,
            'ops': ops,}
    return arch


class Nb201Dataset(Dataset):
    def __init__(self, split, candidate_ops=5, data_type='train', data_set='cifar10'):
        self.nasbench201_dict = np.load('datasets/nasbench201/nasbench201_dict_search.npy', allow_pickle=True).item()
        self.sample_range = list()
        self.candidate_ops = candidate_ops
        if data_type == 'train':
            self.sample_range = random.sample(range(0, len(self.nasbench201_dict)), int(split))
        elif data_type == 'valid':
            self.sample_range = random.sample(range(0, len(self.nasbench201_dict)), int(split))
        elif data_type == 'test':
            self.sample_range = range(0, len(self.nasbench201_dict))
        else:
            pass

        self.data_type = data_type
        self.data_set = data_set
        if self.data_set == 'cifar10':
            self.val_mean, self.val_std = 0.836735, 0.128051
            self.test_mean, self.test_std = 0.870563, 0.129361
        elif self.data_set == 'cifar100':
            self.val_mean, self.val_std = 0.612818, 0.121428
            self.test_mean, self.test_std = 0.613878, 0.121719
        elif self.data_set == 'imagenet16':
            self.val_mean, self.val_std = 0.337928, 0.092423
            self.test_mean, self.test_std = 0.335682, 0.095140
        else:
            pass
        self.max_edge_num = 6

    def __len__(self):
        return len(self.sample_range)

    def normalize(self, num):
        if self.data_type == 'train':
            return (num - self.val_mean) / self.val_std
        elif self.data_type == 'test':
            return (num - self.test_mean) / self.test_std
        else:
            pass

    def denormalize(self, num):
        if self.data_type == 'train':
            return num * self.val_std + self.val_mean
        elif self.data_type == 'test':
            return num * self.test_std + self.test_mean
        else:
            pass

    def __getitem__(self, index):
        index = self.sample_range[index]
        val_acc = self.nasbench201_dict[str(index)]['%s_valid' % self.data_set]
        test_acc = self.nasbench201_dict[str(index)]['%s_test' % self.data_set]
        arch_list=self.nasbench201_dict[str(index)]['arch']
        arch=nasbench201_to_nasbench101(arch_list)
        adjacency=arch['adj']
        ops=arch['ops']
        operation=[op_list[i] for i in ops]
        operation=np.array(operation)

        # edges
        edge_index = []
        for i in range(adjacency.shape[0]):
            idx_list = np.where(adjacency[i])[0].tolist()
            for j in idx_list:
                edge_index.append([i, j])
        if np.sum(edge_index) == 0:
            edge_index = []
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[0] - 1, i, -1):
                    edge_index.append([i, j])

        edge_num = len(edge_index)
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.transpose(1, 0)

        result = {
            "num_vertices": len(ops),
            "edge_num": edge_num,
            "adjacency": np.array(adjacency, dtype=np.float32),
            "operations": ops,
            "features": torch.from_numpy(operation).long(),
            "n_val_acc": torch.tensor(self.normalize(val_acc/100), dtype=torch.float32),
            "n_test_acc": torch.tensor(self.normalize(test_acc/100), dtype=torch.float32),
            "val_acc": val_acc/100,
            "test_acc": test_acc/100,
            "edge_index_list": edge_index,
        }
        return result

