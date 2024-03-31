import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader
from DataLoader import GraphDataset
from dgl import backend as B


# PNAConv
class PNAConv_Module(nn.Module):
    def __init__(self, in_size, out_size, edge_feat_size, aggregators, scalers, last = 0, delta = 1.0,
                 dropout = 0.0, num_towers = 1):
       super(PNAConv_Module, self).__init__()
       self.last = last
       self.PNA = dglnn.PNAConv(in_size, out_size, aggregators, scalers, delta, dropout, num_towers, edge_feat_size)

    def forward(self, g, n_feat, e_feat):
        h = self.PNA(g, n_feat, e_feat)
        if self.last == 0:
            h = F.relu(h)
        return h

# Conv1D
class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, if_final = 0):
        super(Conv1D, self).__init__()
        self.if_final = if_final
        self.Conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.Conv(x)
        if self.if_final != 0 :
            x = F.relu(x)
        return x

class Actor_Net_PNAConv_Model(nn.Module):
    def __init__(self, in_node_feats, out_node_feats, in_edge_feats = 2):
        super(Actor_Net_PNAConv_Model, self).__init__()
        aggregators = ['mean', 'max', 'sum', 'std']
        scalars = ['identity']
        self.in_node_feats = in_node_feats
        self.out_node_feats = out_node_feats
        self.PNAConvModule_1 = PNAConv_Module(in_node_feats-1, 64, in_edge_feats, aggregators, scalars)
        self.PNAConvModule_2 = PNAConv_Module(64, 64, in_edge_feats, aggregators, scalars)
        self.PNAConvModule_3 = PNAConv_Module(64, 8, in_edge_feats, aggregators, scalars)

        self.PNAConvModule_01 = PNAConv_Module(12, 64, in_edge_feats, aggregators, scalars)
        self.PNAConvModule_02 = PNAConv_Module(64, 64, in_edge_feats, aggregators, scalars)
        self.PNAConvModule_03 = PNAConv_Module(64, 1, in_edge_feats, aggregators, scalars, last = 1)


    def forward(self, obs, state = None, info = {}):
        GraphDatas = GraphDataset(obs)
        GraphLoader = GraphDataLoader(GraphDatas, batch_size=obs.shape[0])
        GraphLoader_iter = iter(GraphLoader)
        gs = next(GraphLoader_iter)[0]
        
        ns = th.tensor(gs.ndata["NodeAttr"][:,:-1], dtype=th.float32)
        ns_s = th.tensor(gs.ndata["NodeAttr"][:,-1], dtype=th.float32)
        es = th.tensor(gs.edata["EdgeAttr"], dtype=th.float32)
        
        # k = 0
        # for Graphs in GraphLoader:
        #     gs = Graphs
        #     ns = th.tensor(Graphs.ndata["NodeAttr"][:,:-1], dtype=th.float32)
        #     ns_s = th.tensor(Graphs.ndata["NodeAttr"][:,-1], dtype=th.float32)
        #     es = th.tensor(Graphs.edata["EdgeAttr"], dtype=th.float32)
        #     if k == 0: break

        n_feat_1 = self.PNAConvModule_1(gs, ns, es)
        n_feat_2 = self.PNAConvModule_2(gs, n_feat_1, es)
        n_feat_3 = self.PNAConvModule_3(gs, n_feat_2, es)

        ns_s_tile = th.transpose(th.tile(ns_s, [4, 1]), 0, 1)

        t_feat = th.cat((n_feat_3, ns_s_tile), -1)
        t_feat = self.PNAConvModule_01(gs, t_feat, es)
        t_feat = self.PNAConvModule_02(gs, t_feat, es)
        t_feat = self.PNAConvModule_03(gs, t_feat, es)

        seglen = gs.batch_num_nodes(None)

        soft_feat = dgl.ops.segment_softmax(seglen, t_feat)
        action_1 = B.pad_packed_tensor(soft_feat, seglen, 0, l_min=self.out_node_feats).squeeze(-1)

        return action_1, state

class Critic_Net_PNAConv_Model(Actor_Net_PNAConv_Model):
    def __init__(self, in_node_feats, out_node_feats):
        super(Critic_Net_PNAConv_Model, self).__init__(in_node_feats, out_node_feats)
        self.GlobalPooling = dglnn.MaxPooling()
        self.Conv1_C = Conv1D(64, 1, if_final = 1)

    def forward(self, obs):
        GraphDatas = GraphDataset(obs)
        GraphLoader = GraphDataLoader(GraphDatas, batch_size=obs.shape[0])

        GraphLoader_iter = iter(GraphLoader)
        gs = next(GraphLoader_iter)[0]
        
        ns = th.tensor(gs.ndata["NodeAttr"][:,:-1], dtype=th.float32)
        ns_s = th.tensor(gs.ndata["NodeAttr"][:,-1], dtype=th.float32)
        es = th.tensor(gs.edata["EdgeAttr"], dtype=th.float32)
        
        # k = 0
        # for Graphs in GraphLoader:
        #     gs = Graphs
        #     ns = th.tensor(Graphs.ndata["NodeAttr"][:, :-1], dtype=th.float32)
        #     ns_s = th.tensor(Graphs.ndata["NodeAttr"][:, -1], dtype=th.float32)
        #     es = th.tensor(Graphs.edata["EdgeAttr"], dtype=th.float32)
        #     if k == 0: break

        n_feat_1 = self.PNAConvModule_1(gs, ns, es)
        n_feat_2 = self.PNAConvModule_2(gs, n_feat_1, es)
        n_feat_3 = self.PNAConvModule_3(gs, n_feat_2, es)

        ns_s_tile = th.transpose(th.tile(ns_s, [4, 1]), 0, 1)

        t_feat = th.cat((n_feat_3, ns_s_tile), -1)
        t_feat = self.PNAConvModule_01(gs, t_feat, es)
        t_feat = self.PNAConvModule_02(gs, t_feat, es)

        g_feat = self.GlobalPooling(gs, t_feat)
        g_feat = g_feat.unsqueeze(-1)
        logits = self.Conv1_C(g_feat)

        return logits



