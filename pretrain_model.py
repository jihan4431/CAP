import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
from nb_dataset_101 import Nb101Dataset
from nb_dataset_201 import Nb201Dataset
from darts_dataset import Darts_dataset
import random
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import torch.optim as optim
from argparse import ArgumentParser
from model import GINConv,GNN
def arch2data(arch):
    x=arch["features"]
    edge_num=arch["edge_num"]
    edge_index = arch["edge_index_list"][:,:edge_num]
    verse_edge_index=torch.cat(( edge_index[1,:].unsqueeze(0), edge_index[0,:].unsqueeze(0)),dim=0)
    edge_index=torch.cat((edge_index,verse_edge_index),dim=1)
    pyg_data = Data(x, edge_index).contiguous()
    return pyg_data
def nx_to_graph_data_obj_simple(G):

    arch_features_list = []
    for _, node in G.nodes(data=True):
        arch_feature = node['arch_num_idx']
        arch_features_list.append(arch_feature)
    x = torch.tensor(np.array(arch_features_list), dtype=torch.long)

    if len(G.edges()) > 0:
        edges_list = []

        for i, j, edge in G.edges(data=True):
            edges_list.append((i, j))
            edges_list.append((j, i))
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data

def graph_data_obj_to_nx_simple(data):
    G = nx.Graph()
    arch_features = data.x.cpu().numpy()
    arch_nums = arch_features.shape[0]
    for i in range(arch_nums):
        arch_num_idx = arch_features[i]
        G.add_node(i, arch_num_idx=arch_num_idx)
        pass

    edge_index = data.edge_index.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)
    return G

def reset_idxes(G):
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping

class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        self.k = k
        self.l1 = l1
        self.l2 = l2

        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        num_archs = data.x.size()[0]
        if not data.x[-1].any():
            num_archs-=1
        if root_idx == None:
            root_idx = random.sample(range(num_archs), 1)[0]

        G = graph_data_obj_to_nx_simple(data)
        substruct_node_idxes = nx.single_source_shortest_path_length(G,root_idx,self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[root_idx]])
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_index_context = context_data.edge_index

        context_substruct_overlap_idxes = list(set(context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]  for old_idx in context_substruct_overlap_idxes]
            data.overlap_context_substruct_idx =torch.tensor(context_substruct_overlap_idxes_reorder)
        return data

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)

class BatchSubstructContext(Data):
    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        batch = BatchSubstructContext()  #
        keys = ["center_substruct_idx", "edge_index_substruct", "x_substruct",
                "overlap_context_substruct_idx", "edge_index_context", "x_context"]
        for key in keys:
            batch[key] = []
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0
        i = 0
        for data in data_list:
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batch.batch_overlapped_context.append(
                    torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                for key in ["center_substruct_idx", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                for key in ["overlap_context_substruct_idx", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)
        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx",
                       "center_substruct_idx"]

    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1

class DataLoaderSubstructContext(DataLoader):
    def __init__(self, data_list, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            data_list,
            batch_size,
            shuffle,
            collate_fn=lambda data_list:BatchSubstructContext.from_data_list(data_list),**kwargs)


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def train(model, context_model, loader, optimizer_substruct, optimizer_context, criterion, device):
    model.train()
    context_model.train()
    balanced_loss_accum = 0
    acc_accum = 0
    pos_acc=0
    neg_acc=0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        substruct_rep = model(batch.x_substruct,batch.edge_index_substruct)[batch.center_substruct_idx]
        overlapped_node_rep = context_model(batch.x_context,batch.edge_index_context)[batch.overlap_context_substruct_idx]
        expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i], 1))
                                            for i in range(len(substruct_rep))], dim=0)
        pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim=1)
        shifted_expanded_substruct_rep = []
        neg_samples = 1
        for i in range(neg_samples):
            shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), i + 1)]
            shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat(
                (batch.overlapped_context_size[i], 1)) for i in range(len(shifted_substruct_rep))], dim=0))
        shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim=0)
        pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((neg_samples, 1)), dim=1)
        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()
        loss = loss_pos + neg_samples * loss_neg
        loss.backward()
        optimizer_substruct.step()
        optimizer_context.step()
        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += 0.5 * (float(torch.sum(pred_pos > 0).detach().cpu().item()) / len(pred_pos)
                            + float(torch.sum(pred_neg < 0).detach().cpu().item()) / len(pred_neg))
        pos_acc+=float(torch.sum(pred_pos > 0).detach().cpu().item()) / len(pred_pos)
        neg_acc+=float(torch.sum(pred_neg < 0).detach().cpu().item()) / len(pred_neg)

    return balanced_loss_accum / (step+1), acc_accum / (step+1)

def get_params():
    parser = ArgumentParser()
    # exp and dataset
    parser.add_argument("--exp_name", type=str, default='pretrain')
    parser.add_argument("--bench", type=str, default='101',choices=['101','201','DARTS'])
    parser.add_argument("--split", type=int, default=381262) #90% nb101:381262 nb201:14063 DARTS:1000000
    parser.add_argument("--dataset", type=str, default='cifar10',choices=['cifar10','cifar100','imagenet16'])
    # training settings
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--epochs", default=300) #500 for 101, 300 for 201/DARTS
    parser.add_argument("--batch_size",default=1024,type=int)
    parser.add_argument("--main_lr", default=1e-3, type=float)
    parser.add_argument("--main_wd", default=1e-4, type=float)
    parser.add_argument("--aux_lr", default=1e-3, type=float)
    parser.add_argument("--aux_wd", default=1e-4, type=float)
    parser.add_argument("--K", default=1, type=int)
    parser.add_argument("--R", default=2, type=int)

    args , _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    params = vars(get_params())
    #seed
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    #device
    device = torch.device(torch.device('cuda:' + str(params['gpu'])) if torch.cuda.is_available() else torch.device('cpu'))
    #dataset
    if params['bench'] == '101':
        dataset = Nb101Dataset(split=params['split'],datatype='train')
        model = GNN(3, 128)
        model = model.to(device)
        context_model = GNN(2, 128)
        context_model = context_model.to(device)

    if params['bench'] == '201':
        dataset = Nb201Dataset(split=params['split'], data_type='train',data_set=params['dataset'])
        model = GNN(3, 128,num_op_type=7)
        model = model.to(device)
        context_model = GNN(2, 128,num_op_type=7)
        context_model = context_model.to(device)

    if params['bench'] == 'DARTS':
        dataset = Darts_dataset(split=params['split'], datatype='test')
        model = GNN(3, 128, num_op_type=11)
        model = model.to(device)
        context_model = GNN(2, 128, num_op_type=11)
        context_model = context_model.to(device)

    data_list=[arch2data(data) for data in dataset]
    datalist=[]
    for data in data_list:
        single_SCP=ExtractSubstructureContextPair(params['K'],0,params['R'])
        new_data=single_SCP(data)
        datalist.append(new_data)

    print("start loading.....")
    loader = DataLoaderSubstructContext(datalist, batch_size=params['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    optimizer_substruct = optim.Adam(model.parameters(), lr=params['main_lr'], weight_decay=params['main_wd'])
    optimizer_context = optim.Adam(context_model.parameters(), lr=params['aux_lr'], weight_decay=params['aux_wd'])
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = params['epochs']
    log_loss = []
    log_acc = []
    model = model.to(device)
    context_model = context_model.to(device)
    best_acc = -1
    for epoch in range(epochs):
        print("====epoch " + str(epoch))
        train_loss, train_acc = train(model, context_model, loader, optimizer_substruct, optimizer_context, criterion, device)
        log_loss.append(train_loss)
        log_acc.append(train_acc)
        split = str(params['split'])
        np.save("pth/"+params['bench']+'/'+split+"Context_Pretrain_loss.npy", log_loss)
        np.save("pth/"+params['bench']+'/'+split+"Context_Pretrain_log_acc.npy", log_acc)
        print('Epoch:{},loss:{}, acc:{}'.format(epoch, train_loss, train_acc))
        torch.save(model.state_dict(), "pth/"+params['bench']+'/'+split+"Context_Pretrain_GIN_para.pth")
        torch.save(model, "pth/"+params['bench']+'/'+split+"Context_Pretrain_GIN.pth")
        torch.save(context_model, "pth/"+params['bench']+'/'+split+'context_model.pth')
        if best_acc<train_acc:
            best_acc=train_acc
            torch.save(model.state_dict(), "pth/"+params['bench']+'/'+split+"best_Context_Pretrain_GIN_para.pth")
            torch.save(model, "pth/"+params['bench']+'/'+split+"best_Context_Pretrain_GIN.pth")
            torch.save(context_model, "pth/"+params['bench']+'/'+split+'best_context_model.pth')
        if epoch % 100==0:
            torch.save(model.state_dict(), "pth/"+params['bench']+'/'+split+"Context_Pretrain_GIN_para_"+str(epoch)+".pth")
            torch.save(model,"pth/"+ params['bench']+'/'+split+"Context_Pretrain_GIN_"+str(epoch)+".pth")
            torch.save(context_model, "pth/"+params['bench']+'/'+split+"context_model_"+str(epoch)+".pth")

