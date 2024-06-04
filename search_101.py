import random, copy, time, argparse
import torch
import torch.nn as nn
from nasbench import api as NASBench101API
import numpy as np
from scipy import stats
from nb_dataset_101 import Nb101Dataset
from torch_geometric.data import Data,Batch
import torch.optim as optim
from train_models import GNN_graphpred, BPRLoss, GNN, Predictor,AvgrageMeter,GINConv
from torch.utils.data import DataLoader
from functools import cmp_to_key
import  collections
class NAS(object):
    def __init__(self, N, search_space, dataset, flops_limit,api_loc=None,
                 graph_pred_model_state_dict=None, device='cpu', seed=None):
        self.N = N
        self.search_space = search_space
        self.dataset = dataset
        self.flops_limit = flops_limit
        self.device = device
        self.seed = seed
        self.loss = args.loss
        self.visited = []
        if self.search_space == 'nasbench101':
            self.nasbench = NASBench101API.NASBench(api_loc)
            self.available_ops = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']
            self.max_num_vertices = 7
            self.max_num_edges = 9
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        self.pre_model = GNN(3, 128)
        self.linear_model = Predictor(feature_dim=128).to(device)
        self.graph_pred_model = GNN_graphpred(pre_model=self.pre_model, pre_model_files=args.pretrained_model,
                                              graph_pred_linear=self.linear_model, if_pretrain=args.is_pretrained,
                                              drop_ratio=0.15).to(self.device)
        if graph_pred_model_state_dict:
             self.graph_pred_model.load_state_dict(torch.load(graph_pred_model_state_dict))

    def sample_arch(self):
        if self.search_space == 'nasbench101':
            hash_list = list(self.nasbench.hash_iterator())
            hash_value = random.choice(hash_list)
            fixed_statistic = self.nasbench.fixed_statistics[hash_value]
            sampled_arch = (hash_value, fixed_statistic['module_adjacency'], fixed_statistic['module_operations'])
        else:
            raise ValueError('No implementation!')
        return sampled_arch

    def eval_arch(self, arch, use_val_acc=False, model=None):
        start_time = time.time()
        if use_val_acc:
            if self.search_space == 'nasbench101':
                info = self.nasbench.computed_statistics[arch[0]][108]#[np.random.randint(3)]
                val_acc = info[0]['final_validation_accuracy']
                test_acc = (info[0]['final_test_accuracy']+info[1]['final_test_accuracy']+info[2]['final_test_accuracy'])/3
                total_eval_time = info[0]['final_training_time']
                return val_acc, test_acc, total_eval_time
            else:
                raise ValueError('Arch in {:} search space have to be trained or this space does not exist.'.format(
                    self.search_space))
        else:
            model.eval()
            eval_one = Nb101Dataset(split=1, datatype='test', no_sample=True, hash_list=[(arch[0])]).__getitem__(0)
            pyg_data=Data(x=eval_one["features"].clone().detach().to(self.device),
                                edge_index=eval_one["edge_index_list"].clone().detach().to(self.device))
            datalist=[pyg_data]
            pyg_batch=Batch.from_data_list(datalist).to(self.device)
            with torch.no_grad():
                if model:
                    output = model(pyg_batch)
                else:
                    output = self.graph_pred_model(pyg_batch)
                output = output.detach()
                measure = output.squeeze().detach().cpu().item()
            total_eval_time = time.time() - start_time
            return measure, total_eval_time, output

    def cmp(self, x, y):
        ret = x[1] - y[1]
        if ret < 0:
            return -1
        elif ret > 0:
            return 1
        else:
            return 0

    def train(self, history, model, epochs):
        batch_size = (len(history) - 1) // 2 + 1
        archs_hash = []
        ars=[]
        for h in history:
            if self.search_space == 'nasbench101':
                archs_hash.append(h[0][0])
                ars.append(h[0])
            else:
                raise ValueError('There is no {:} search space.'.format(self.search_space))

        train_set=Nb101Dataset(split=len(history),datatype='train',no_sample=True,hash_list=archs_hash)

        train_loader=DataLoader(train_set,batch_size=batch_size,num_workers=0,shuffle=True)#,drop_last=True)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs,eta_min=1e-4)
        if self.loss == 'mse':
            criterion = nn.MSELoss()
        if self.loss == 'bpr':
            criterion = BPRLoss()
        iters = range(1, epochs+ 1)
        for _ in iters:
            losses = AvgrageMeter()
            model.train()
            optimizer.zero_grad()
            targets=[]
            outputs=[]
            for step,batch in enumerate(train_loader):
                target=batch["val_acc"].clone().detach().to(self.device)
                n=target.size(0)
                data_list=[]
                for i in range(n):
                    s_data=Data(x=batch["features"][i,:].clone().detach().to(self.device),
                                edge_index=batch["edge_index_list"][i,:,:].clone().detach().to(self.device))
                    data_list.append(s_data)
                pyg_batch=Batch.from_data_list(data_list)
                optimizer.zero_grad()
                output=model(pyg_batch)
                loss = criterion(output.squeeze(),target.squeeze())
                loss.backward()
                optimizer.step()
                losses.update(loss.data.item(),n)
                outputs.append(output.squeeze().detach())
                targets.append(target.squeeze().detach())

            outputs=torch.cat(outputs).cpu().numpy()
            targets=torch.cat(targets).cpu().numpy()
            tau=stats.kendalltau(targets,outputs,nan_policy='omit')[0]
            train_loss,train_tau=losses.avg,tau
            scheduler.step()
            if _== epochs :print("epoch:",_,"loss:",train_loss,"tau:",train_tau)
        return model
    def predict(self, candidates, model):
        predictions = []
        for c in candidates:
            measure, _, _ = self.eval_arch(c, use_val_acc=False, model=model)
            predictions.append(measure)
        return predictions

class Random_NAS(NAS):
    def __init__(self, N, search_space, dataset, flops_limit, api_loc=None , num_init_archs=20,device='cpu', K=10, seed=None):
        super(Random_NAS, self).__init__(N, search_space, dataset, flops_limit, api_loc=api_loc, device=device, seed=seed)
        self.num_init_archs = num_init_archs
        self.K = K
    def get_candidates(self):
        patience_factor = 5
        num = 100
        candidates = []
        for _ in range(patience_factor):
            for _ in range(int(num)):
                arch = self.sample_arch()
                if arch[0] not in self.visited:
                    candidates.append(arch)
                    self.visited.append(arch[0])
                if len(candidates) >= num:
                    return candidates
        return candidates

    def run(self):
        total_eval_time = 0
        history  = []
        while len(history) < self.num_init_archs:
            arch = self.sample_arch()
            if arch[0] not in self.visited:
                valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                cur = (arch, valid_acc, test_acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
                self.visited.append(arch[0])
        print("len:", len(history))
        print('best:', max(history, key=lambda x: x[2]))
        while len(history) < self.N:
            candidates = self.get_candidates()
            graph_pred_model = copy.deepcopy(self.graph_pred_model)
            graph_pred_model = self.train(history, graph_pred_model,epochs=args.epochs)
            candidate_predictions=self.predict(candidates, graph_pred_model)
            candidate_indices = np.argsort(candidate_predictions)

            for i in candidate_indices[-self.K:]:
                arch = candidates[i]
                valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                cur = (arch, valid_acc, test_acc, eval_time)
                total_eval_time += eval_time
                history.append(cur)
            print("len:",len(history))
            print('best:',max(history, key=lambda x: x[2]))
        best = max(history, key=lambda x: x[2])
        history.sort(key=lambda x:x[2],reverse=True)
        return best, history, total_eval_time

class Evolved_NAS(NAS):

    def __init__(self, N, search_space, population_size, tournament_size, dataset, flops_limit, api_loc=None, K=10, device='cpu', seed=None):
        super(Evolved_NAS, self).__init__(N, search_space, dataset, flops_limit, api_loc=api_loc, device=device, seed=seed)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.K = K

    def mutate(self, parent, p):
        if self.search_space == 'nasbench101':
            if random.random() < p:
                while True:
                    old_matrix, old_ops = parent[1], parent[2]
                    idx_to_change = random.randrange(len(old_ops[1:-1])) + 1
                    entry_to_change = old_ops[idx_to_change]
                    possible_entries = [x for x in self.available_ops[1:-1] if x != entry_to_change]
                    new_entry = random.choice(possible_entries)
                    new_ops = copy.deepcopy(old_ops)
                    new_ops[idx_to_change] = new_entry
                    idx_to_change = random.randrange(sum(range(1, len(old_matrix))))
                    new_matrix = copy.deepcopy(old_matrix)
                    num_node = len(old_matrix)
                    idx_to_ij = {int(i*(num_node-1)-i*(i-1)/2+(j-i-1)): (i, j) for i in range(num_node) for j in range(i+1, num_node)}
                    i, j = idx_to_ij[idx_to_change]
                    new_matrix[i][j] = 1 if new_matrix[i][j] == 0 else 0
                    new_spec = NASBench101API.ModelSpec(matrix=new_matrix, ops=new_ops)
                    if self.nasbench.is_valid(new_spec):
                        spec_hash = new_spec.hash_spec(self.available_ops[1:-1])
                        child = (spec_hash, new_matrix, new_ops)
                        break
            else:
                child = parent
        else:
            raise ValueError('There is no {:} search space.'.format(self.search_space))
        return child
    def get_candidates(self, arch_pool):
        p = 1.0
        num_arches_to_mutate = 1
        patience_factor = 5
        num = 100
        candidates = []
        for _ in range(patience_factor):
            samples  = random.sample(arch_pool, self.tournament_size)
            parents = [arch[0] for arch in sorted(samples, key=cmp_to_key(self.cmp), reverse=True)[:num_arches_to_mutate]]
            for parent in parents:
                for _ in range(int(num / num_arches_to_mutate)):
                    child = self.mutate(parent, p)
                    if child[0] not in self.visited:
                        candidates.append(child)
                        self.visited.append(child[0])
                    if len(candidates) >= num:
                        return candidates
        return candidates

    def run(self):
        total_eval_time = 0
        history  = []
        population = collections.deque()
        while len(history) < self.population_size:
            arch = self.sample_arch()
            if arch[0] not in self.visited:
                valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                cur = (arch, valid_acc, test_acc, eval_time)
                population.append(cur)
                history.append(cur)
                self.visited.append(arch[0])
        print('len:',len(history))
        print('best:', max(history, key=lambda x: x[2]))
        while len(history) < self.N:
            candidates = self.get_candidates(population)
            graph_pred_model=copy.deepcopy(self.graph_pred_model)
            graph_pred_model=self.train(history,graph_pred_model,args.epochs)
            candidate_predictions = self.predict(candidates, graph_pred_model)
            candidate_indices = np.argsort(candidate_predictions)
            for i in candidate_indices[-self.K:]:
                arch = candidates[i]
                valid_acc, test_acc, eval_time = self.eval_arch(arch, use_val_acc=True)
                cur = (arch, valid_acc, test_acc, eval_time)
                total_eval_time += eval_time
                population.append(cur)
                history.append(cur)
                population.popleft()
            print("len:", len(history))
            print('best:', max(history, key=lambda x: x[2]))
        history.sort(key=lambda x:x[2],reverse=True)
        best = max(history, key=lambda x: x[2])
        return best, history, total_eval_time

parser = argparse.ArgumentParser(description='Neural Architecture Search')
parser.add_argument('--search_space', default='nasbench101', type=str)
parser.add_argument('--search_algo', default='rea', type=str, choices=['r', 'rea'])
parser.add_argument('--dataset', default='cifar10',type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--N', default=150, type=int)
parser.add_argument('--population_size', default=20, type=int)
parser.add_argument('--tournament_size', default=5, type=int)
parser.add_argument('--EMA_momentum', default=0.9, type=float)
parser.add_argument('--flops_limit', default=600e6, type=float)
#######
parser.add_argument('--is_pretrained', default=True, type=bool)
parser.add_argument("--pretrained_model", default='wts/Pretrain_101.pth', type=str)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--wd", default=1e-3, type=float)
parser.add_argument("--dropout", default=0.15, type=float)
parser.add_argument("--loss", type=str, default='bpr', choices=['mse', 'bpr'])
parser.add_argument("--epochs", default=250, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    args.api_loc='datasets/nasbench101/nasbench_full.tfrecord'
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.search_algo == 'r':
        nas= Random_NAS(args.N, args.search_space, args.dataset, args.flops_limit, api_loc=args.api_loc,device=device,seed=args.seed)
    elif args.search_algo == 'rea':
        nas=Evolved_NAS(args.N, args.search_space, args.population_size, args.tournament_size,args.dataset, args.flops_limit, api_loc=args.api_loc,device=device,seed=args.seed)

    begin_time = time.time()
    best, history, total_eval_time = nas.run()
    end_time = time.time()
    print("==========best======")
    print(best)




