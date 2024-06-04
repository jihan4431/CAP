from tqdm import tqdm
import numpy as np
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from nb_dataset_101 import Nb101Dataset
from nb_dataset_201 import Nb201Dataset
from torch.utils.data import DataLoader
from scipy import stats
from torch_geometric.data import Data, Batch
from argparse import ArgumentParser
import logging
from utils import AvgrageMeter,BPRLoss
from model import GNN,GINConv,Predictor,GNN_graphpred

parser = ArgumentParser()
# exp and dataset
parser.add_argument("--exp_name", type=str, default='rank')
parser.add_argument("--bench", type=str, default='101')
parser.add_argument("--train_split", type=int, default=100)
parser.add_argument("--test_split", type=str, default='all')
parser.add_argument("--dataset", type=str, default='cifar10')
# training settings
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--loss", type=str, default='bpr',choices=['mse','bpr'])
parser.add_argument("--epochs", default=250, type=int)
parser.add_argument("--layers", default=3, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--wd", default=1e-3, type=float)
parser.add_argument("--dropout", default=0.15, type=float)
parser.add_argument("--train_batch_size", default=10, type=int)
parser.add_argument("--test_batch_size", default=10240, type=int)
parser.add_argument("--is_pretrained", default=True, type=bool)
parser.add_argument("--pretrained_model", default='wts/Pretrain_101.pth', type=str,choices=['wts/Pretrain_101.pth','wts/Pretrain_201.pth','wts/Pretrain_DARTS.pth'])
args = parser.parse_args()



def train(loader, model, optimizer, criterion):
    losses = AvgrageMeter()
    batch_times = AvgrageMeter()
    device = torch.device(torch.device('cuda:'+str(args.gpu)) if torch.cuda.is_available() else torch.device('cpu'))
    model.train()
    optimizer.zero_grad()
    outputs = []
    targets = []
    for step, batch in enumerate(loader):
        target = batch["n_val_acc"].clone().detach().cuda(non_blocking=True).to(device)
        n = target.size(0)
        data_list = []
        for i in range(n):
            s_data = Data(x=batch["features"][i, :].clone().detach().cuda(non_blocking=True).to(device),
                          edge_index=batch["edge_index_list"][i, :, :].clone().detach().cuda(non_blocking=True).to(device))
            data_list.append(s_data)
        pyg_batch = Batch.from_data_list(data_list)
        b_start = time.time()
        optimizer.zero_grad()
        output = model(pyg_batch)
        if criterion == "mse":
            target = batch["n_val_acc"].clone().detach().float().cuda(non_blocking=True).to(device)
            loss_func = nn.MSELoss()
            loss = loss_func(output.squeeze(), target.squeeze().float()).float()
        if criterion == "bpr":
            loss_func = BPRLoss()
            loss = loss_func(output.squeeze(), target.squeeze())
        loss.backward()
        optimizer.step()
        batch_times.update(time.time() - b_start)
        losses.update(loss.data.item(), n)
        outputs.append(output.squeeze().detach())
        targets.append(target.squeeze().detach())

    outputs = torch.cat(outputs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    tau = stats.kendalltau(targets, outputs, nan_policy='omit')[0]
    return losses.avg, tau, batch_times.avg


def test(all_loader, model, criterion):
    device=torch.device(torch.device('cuda:'+str(args.gpu)) if torch.cuda.is_available() else torch.device('cpu'))
    all_losses = AvgrageMeter()
    all_batch_times = AvgrageMeter()
    model.eval()
    outputs = []
    targets = []
    for step, batch in enumerate(tqdm(all_loader)):
        target = batch["test_acc"].clone().detach().cuda(non_blocking=True).to(device)
        n = target.size(0)
        data_list = []
        for i in range(n):
            s_data = Data(x=batch["features"][i, :].clone().detach().cuda(non_blocking=True).to(device),
                          edge_index=batch["edge_index_list"][i, :, :].clone().detach().cuda(non_blocking=True).to(device))
            data_list.append(s_data)
        pyg_batch = Batch.from_data_list(data_list)
        start = time.time()
        with torch.no_grad():
            output= model(pyg_batch)
            output = output.detach()
        if criterion == "mse":
            target = batch["n_test_acc"].clone().detach().float().cuda(non_blocking=True).to(device)
            loss_func = nn.MSELoss()
            loss = loss_func(output.squeeze(), target.squeeze()).float()
        if criterion == "bpr":
            loss_func = BPRLoss()
            loss = loss_func(output.squeeze(), target.squeeze())
        all_batch_times.update(time.time() - start)
        all_losses.update(loss.data.item(), n)
        outputs.append(output.squeeze().detach())
        targets.append(target.squeeze().detach())
    outputs = torch.cat(outputs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    all_tau = stats.kendalltau(targets, outputs, nan_policy='omit')[0]
    return all_losses.avg, all_tau, all_batch_times.avg


if __name__ == '__main__':
    # initialize log info
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.info(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(torch.device('cuda:'+str(args.gpu)) if torch.cuda.is_available() else torch.device('cpu'))
    epochs = args.epochs
    if args.bench == '101':
        train_set = Nb101Dataset(split=args.train_split, datatype='train')
        test_set = Nb101Dataset(split=args.test_split, datatype='test')
        #eval_set= Nb101Dataset(split=200, datatype='test')
        pre_model = GNN(args.layers, 128).to(device)
    if args.bench == '201':
        train_set=Nb201Dataset(split=args.train_split,data_type='train')
        test_set=Nb201Dataset(split=args.test_split,data_type='test')
        # eval_set= Nb201Dataset(split=200, data_type='test')
        pre_model = GNN(args.layers, 128, num_op_type=7).to(device)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=0, shuffle=True, drop_last=True)
    # eval_loader = DataLoader(eval_set, batch_size=20, shuffle=False,num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=6)
    linear_model = Predictor(feature_dim=128).to(device)
    criterion = args.loss
    model = GNN_graphpred(pre_model=pre_model, pre_model_files=args.pretrained_model,
                                       graph_pred_linear=linear_model, if_pretrain=args.is_pretrained, drop_ratio=args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for ep in range(1, 1 + epochs):
        loss, tau, batch_time= train(loader=train_loader,model=model,optimizer=optimizer,criterion=criterion)
        print("epoch:", ep,"loss.avg:", loss, "tau:", tau, "batch_time:", batch_time)

    test_loss, test_tau, test_batch_time=test(all_loader=test_loader,model=model,criterion=criterion)
    print("=====test==========")
    print("loss.avg:", test_loss, "test_tau:", test_tau, "val-time:", test_batch_time)



