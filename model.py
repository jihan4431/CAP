import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,global_add_pool,global_mean_pool,global_max_pool
from torch_geometric.utils import add_self_loops
class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.aggr = aggr

    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        edge_index = edge_index.long()

        return self.propagate(edge_index=edge_index, x=x)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_op_type=5, JK="last", drop_ratio=0.5):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_op_type, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, aggr="add"))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))


    def forward(self, *argv):
        if len(argv) == 2:
            x, edge_index = argv[0], argv[1]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index = data.x, data.edge_index
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x)
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        return node_representation

class Predictor(nn.Module):
    def __init__(self, feature_dim):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.PReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.PReLU(),
            nn.Linear(feature_dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, z):
        return self.fc(z)

class GNN_graphpred(torch.nn.Module):
    def __init__(self, pre_model, pre_model_files, graph_pred_linear, drop_ratio=0.05, graph_pooling="mean",
                 if_pretrain=True):
        super(GNN_graphpred, self).__init__()
        self.drop_layer = torch.nn.Dropout(p=drop_ratio)
        self.gnn = pre_model
        self.pre_model_files = pre_model_files
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        self.pred_linear = graph_pred_linear
        if if_pretrain:
            self.from_pretrained()

    def from_pretrained(self, ):
        self.gnn = torch.load(self.pre_model_files)
        self.gnn = self.gnn.eval()

    def forward(self, data):
        batch = data.batch
        node_representation = self.gnn(data)
        result = self.pool(node_representation, batch)
        result = self.drop_layer(result)
        result = self.pred_linear(result)
        return result
