"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class RGCN(nn.Module):
    def __init__(self, parameter, is_module=True):
        super(RGCN, self).__init__()

        self.inp_dim = parameter['embed_dim']
        self.emb_dim = parameter['rgcn_embed_dim']
        self.attn_rel_emb_dim = parameter['rgcn_embed_dim']
        self.num_rels = parameter['num_relations']
        self.num_bases = parameter['num_bases']
        self.num_hidden_layers = parameter['num_gcn_layers']
        self.dropout = parameter['dropout']
        self.edge_dropout = parameter['edge_dropout']
        self.neighborhood_sample_rate = parameter['neighborhood_sample_rate']
        self.has_attn = parameter['has_attn']
        self.is_module = is_module

        self.device = parameter['device']

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        # initialize aggregators for input and hidden layers
        if parameter['gnn_agg_type'] == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif parameter['gnn_agg_type'] == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif parameter['gnn_agg_type'] == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNBasisLayer(self.inp_dim,
                              self.emb_dim,
                              self.aggregator,
                              self.attn_rel_emb_dim,
                              self.num_rels,
                              self.num_bases,
                              activation=F.relu,
                              dropout=self.dropout,
                              edge_dropout=self.edge_dropout,
                              is_input_layer=True,
                              has_attn=self.has_attn,
                              attn_rel_emb=self.attn_rel_emb)

    def build_hidden_layer(self, idx):
        return RGCNBasisLayer(self.emb_dim,
                              self.emb_dim,
                              self.aggregator,
                              self.attn_rel_emb_dim,
                              self.num_rels,
                              self.num_bases,
                              activation=F.relu,
                              dropout=self.dropout,
                              edge_dropout=self.edge_dropout,
                              has_attn=self.has_attn,
                              attn_rel_emb=self.attn_rel_emb)

    def forward(self, batch_nodes, g):
        g.readonly(True)
        for j, nf in enumerate(dgl.contrib.sampling.NeighborSampler(g, batch_size=len(batch_nodes),
                                                                    expand_factor=self.neighborhood_sample_rate,
                                                                    num_hops=self.num_hidden_layers,
                                                                    seed_nodes=batch_nodes)):
            nf.copy_from_parent()
            for i, layer in enumerate(self.layers):
                nf.prop_flow(message_funcs=layer.msg_func,
                             reduce_funcs=layer.aggregator,
                             apply_node_funcs=layer.node_update,
                             flow_range=slice(i, self.num_hidden_layers, 1))

        if not self.is_module:
            nf.layers[-1].data['repr'] = nf.layers[-1].data['h'].detach()
            nf.copy_to_parent(node_embed_names=[[], [], ['repr']], edge_embed_names=None)

        return nf.layers[-1].data.pop('h'), nf.layer_parent_nid(-1)


class RGCNBasisLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False, attn_rel_emb=None):
        super(RGCNBasisLayer, self).__init__()

        self.bias = bias
        self.activation = activation

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = nn.Identity()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        if self.has_attn:
            self.attn_rel_emb = attn_rel_emb
            self.A = nn.Linear(2 * self.inp_dim + self.attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)

        w = weight.index_select(0, edges.data['type'])

        input_ = 'feat' if self.is_input_layer else 'h'

        msg = self.edge_dropout(torch.ones(len(edges), 1).to(self.weight.device)) * torch.bmm(edges.src[input_].unsqueeze(1), w).squeeze(1)
        curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (B, F)

        if self.has_attn:
            e = torch.cat([edges.src[input_], edges.dst[input_], self.attn_rel_emb(edges.data['type'])], dim=1)
            a = torch.sigmoid(self.B(F.relu(self.A(e))))
        else:
            a = torch.ones((len(edges), 1)).to(device=w.device)

        return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

    def node_update(self, node):

        # apply bias and activation
        node_repr = node.data['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        # if self.dropout:
        #     node_repr = self.dropout(node_repr)

        node.data['h'] = node_repr

    # def forward(self, g, attn_rel_emb=None):

    #     self.propagate(g, attn_rel_emb)

    #     # apply bias and activation
    #     node_repr = g.ndata['h']
    #     if self.bias:
    #         node_repr = node_repr + self.bias
    #     if self.activation:
    #         node_repr = self.activation(node_repr)
    #     # if self.dropout:
    #     #     node_repr = self.dropout(node_repr)

    #     g.ndata['h'] = node_repr

    #     if self.is_input_layer:
    #         g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
    #     else:
    #         g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)


class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb
