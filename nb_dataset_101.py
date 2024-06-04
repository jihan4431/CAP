import h5py
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset


class Nb101Dataset(Dataset):
    def __init__(self, split=None, debug=False, candidate_ops=5,datatype='pretrain',no_sample=False,hash_list=None):
        self.hash2id = dict()
        with h5py.File("datasets/nasbench101/nasbench.hdf5", mode="r") as f:
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f["num_vertices"][()]
            self.trainable_parameters = f["trainable_parameters"][()]
            self.adjacency = f["adjacency"][()]
            self.operations = f["operations"][()]
            self.metrics = f["metrics"][()]
        self.random_state = np.random.RandomState(0)
        if split is not None and split != "all":
            if no_sample==True:
                self.sample_range=[self.hash2id[val] for val in hash_list]
            else:
                self.sample_range = np.load("datasets/nasbench101/train_samples.npz")[str(split)]
        else:
            self.sample_range = list(range(len(self.hash2id)))
        self.debug = debug
        self.idx = 0
        self.candidate_ops = candidate_ops
        self.data_type=datatype

        # all
        self.val_mean_mean, self.val_mean_std = 0.908192, 0.023961  # all val mean/std (from SemiNAS)
        self.test_mean_mean, self.test_mean_std = 0.8967984, 0.05799569  # all test mean/std

        self.max_edge_num = 9

    def __len__(self):
        return len(self.sample_range)

    def check_norm(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def mean_acc(self):
        return np.mean(self.metrics[:, -1, self.idx, -1, 2])

    def std_acc(self):
        return np.std(self.metrics[:, -1, self.idx, -1, 2])

    def normalize(self, num):
        if self.data_type == 'train':
            return (num - self.val_mean_mean) / self.val_mean_std
        elif self.data_type == 'test':
            return (num - self.test_mean_mean) / self.test_mean_std
        else:
            pass

    def denormalize(self, num):
        if self.data_type == 'train':
            return num * self.val_mean_std + self.val_mean_mean
        elif self.data_type == 'test':
            return num * self.test_mean_std + self.test_mean_mean
        else:
           pass

    def resample_acc(self, index, split="val"):
        # when val_acc or test_acc are out of range
        assert split in ["val", "test"]
        split = 2 if split == "val" else 3
        for seed in range(3):
            acc = self.metrics[index, -1, seed, -1, split]
            if not self._is_acc_blow(acc):
                return acc
        if self.debug:
            print(index, self.metrics[index, -1, :, -1])
            raise ValueError
        return np.array(self.val_mean_mean)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    def _check_validity(self, data_list, threshold=0.3):
        _data_list = deepcopy(data_list).tolist()
        del_idx = []
        for _idx, _data in enumerate(_data_list):
            if _data < threshold:
                del_idx.append(_idx)
        for _idx in del_idx[::-1]:
            _data_list.pop(_idx)
        if len(_data_list) == 0:
            return data_list
        else:
            return np.array(_data_list)

    def __getitem__(self, index):
        index = self.sample_range[index]
        val_acc = self.metrics[index, -1, :, -1, 2]
        test_acc = self.metrics[index, -1, :, -1, 3]
        val_acc = val_acc[0]
        test_acc = self._check_validity(test_acc)
        test_acc = np.mean(test_acc)
        if self._is_acc_blow(val_acc):
            val_acc = self.resample_acc(index, "val")
        n = self.num_vertices[index]
        ops_onehot = np.array([[i == k + 2 for i in range(self.candidate_ops)]
                               for k in self.operations[index]], dtype=np.float32)
        if n < 7:
            ops_onehot[n:] = 0.
        features = np.expand_dims(np.array([i for i in range(self.candidate_ops)]), axis=0)
        features = np.tile(features, (len(ops_onehot), 1))
        features = ops_onehot * features
        features = np.sum(features, axis=-1)
        adjacency = self.adjacency[index]

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
        pad_num = self.max_edge_num - edge_num
        if pad_num > 0:
            edge_index = np.pad(np.array(edge_index), ((0, pad_num), (0, 0)),'constant', constant_values=(0, 0))
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.transpose(1, 0)

        result = {
            "num_vertices": 7,
            "edge_num": edge_num,
            "adjacency": adjacency,
            "operations": ops_onehot,
            "features": torch.from_numpy(features).long(),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "n_val_acc":float(self.normalize(val_acc)),
            "n_test_acc": float(self.normalize(test_acc)),
            "edge_index_list": edge_index,

        }
        if self.debug:
            self._check(result)
        return result


