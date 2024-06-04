from nb_dataset_201 import Nb201Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import torch
from train_models import train
from model import GNN, GNN_graphpred, Predictor,GINConv
import torch.optim as optim
from tqdm import tqdm
from functools import cmp_to_key
import random
import numpy as np
from utils import BPRLoss
from argparse import ArgumentParser

b_size = 1024
def eval_arch(all_loader, model):
    model.eval()
    res = []
    for step, batch in enumerate(tqdm(all_loader)):
        target = batch["test_acc"].clone().detach().cuda(non_blocking=True)
        n = target.size(0)
        data_list = []
        for i in range(n):
            s_data = Data(x=batch["features"][i, :].clone().detach().cuda(non_blocking=True),
                          edge_index=batch["edge_index_list"][i, :, :].clone().detach().cuda(non_blocking=True))
            data_list.append(s_data)
        pyg_batch = Batch.from_data_list(data_list)
        with torch.no_grad():
            output = model(pyg_batch)
            output = output.detach()
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        for i in range(n):
            res.append((i + step * b_size, output[i][0], target[i]))
    return res


parser = ArgumentParser()
# exp and dataset
parser.add_argument("--exp_name", type=str, default='rank')
parser.add_argument("--bench", type=str, default='201')
parser.add_argument("--train_split", type=int, default=50)
parser.add_argument("--num_c", type=int, default=50)
parser.add_argument("--dataset", type=str, default='cifar10')
# training settings
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--loss", type=str, default='bpr',choices=['mse','bpr'])
parser.add_argument("--epochs", default=150, type=int)
parser.add_argument("--layers", default=3, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--wd", default=1e-3, type=float)
parser.add_argument("--dropout", default=0.15, type=float)
parser.add_argument("--train_batch_size", default=5, type=int)
parser.add_argument("--test_batch_size", default=10240, type=int)
parser.add_argument("--is_pretrained", default=True, type=bool)
parser.add_argument("--pretrained_model", default='wts/Pretrain_201.pth', type=str)
args = parser.parse_args()


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(torch.device('cuda:' + str(args.gpu)) if torch.cuda.is_available() else torch.device('cpu'))
    test_set = Nb201Dataset(split='all',data_type='test',data_set=args.dataset)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=6)
    pre_model = GNN(args.layers, 128,num_op_type=7).to(device)
    linear_model = Predictor(feature_dim=128).to(device)
    best_list=[]
    for run in range(10):
        train_set = Nb201Dataset(split=args.train_split, data_type='train',data_set=args.dataset)
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                                  num_workers=0, shuffle=True, drop_last=True)
        model = GNN_graphpred(pre_model=pre_model, pre_model_files=args.pretrained_model,
                                   graph_pred_linear=linear_model, if_pretrain=True, drop_ratio=args.dropout)
        criterion = args.loss
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        print("start training predictor......")
        for epoch in tqdm(range(1, 1 + args.epochs)):
            loss, tau, batch_time = train(loader=train_loader, model=model,
                                             optimizer=optimizer,
                                             criterion=criterion)
            if epoch == args.epochs: print("loss:",loss,"tau:", tau)
        def cmp(a, b):
            ret = a[1] - b[1]
            if ret > 0:
                return -1
            elif ret < 0:
                return 1
            else:
                return 0

        candidate_num = args.num_c
        print("start predicting......")
        predict_list = eval_arch(test_loader, model)
        res_list = [arch for arch in sorted(predict_list, key=cmp_to_key(cmp), reverse=False)[:candidate_num]]
        best = max(res_list, key=lambda k: k[2])
        print("run:",run,"best_arch:",best)
        print(test_set.__getitem__(best[0]))
        best_list.append(best[-1].item())

    best_list=np.array(best_list)
    print("best_archs:",best_list)
    print("mean:",best_list.mean())
    print("std:",best_list.std())


