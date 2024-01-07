import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input

        # node degrees
        degrees = adj.sum(dim=1)

        # Find indices of top 100 nodes with highest degrees
        _, top_nodes = torch.topk(degrees.values(), 100)

        # # Extract the submatrix of adj corresponding to the top nodes
        # sub_adj = adj.to_dense()[top_nodes, :][:, top_nodes]
        # # Find pairs of nodes in the top 100 that are connected
        # connected_pairs = (sub_adj > 0).nonzero(as_tuple=False)

        # Find connections of top 100 nodes with all other nodes
        connected_pairs = []
        connected_values = []
        pair_dict = {}
        for node in top_nodes:
            # Find nodes connected to the current top node
            connected_nodes = ((adj.to_dense())[node] > 0).nonzero(as_tuple=False).squeeze()

            # Create pairs (top node, connected node)
            pairs = torch.stack([node.repeat(connected_nodes.size(0)), connected_nodes], dim=1)
            connected_pairs.append(pairs)

            pair_dict = dict(zip(connected_pairs, [None]*len(connected_pairs)))

            connected_values.append(input[connected_nodes])

        connected_pairs = torch.cat(connected_pairs, dim=0)


        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner


if __name__ == '__main__':
    pass






