from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from .rgcn_model import RGCN


class MLPModule(nn.Module):

    def __init__(self, parameter):
        super(MLPModule, self).__init__()
        self.embedding_dim = parameter['embed_dim']
        self.out_dim = 1

        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(2 * self.embedding_dim, self.embedding_dim)),
            ('bn', nn.BatchNorm1d(parameter['few'])),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=parameter['dropout'])),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.embedding_dim, self.embedding_dim)),
            ('bn', nn.BatchNorm1d(parameter['few'])),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=parameter['dropout'])),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.embedding_dim, self.out_dim)),
            ('bn', nn.BatchNorm1d(parameter['few'])),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, support_set, support_emb, background_graph=None):

        size = support_emb.shape
        x = support_emb.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)
        r_meta = x.view(size[0], self.out_dim)  # dim batch_size, 1
        return r_meta


class RGCNModule(nn.Module):
    def __init__(self, parameter):
        super(RGCNModule, self).__init__()
        self.rgcn_embed_dim = parameter['rgcn_embed_dim']
        self.out_dim = 1
        self.device = parameter['device']

        self.rgcn = RGCN(parameter)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.rgcn_embed_dim, out_features=self.rgcn_embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.rgcn_embed_dim, out_features=self.out_dim),
        )

    def forward(self, support_set, support_emb, background_graph):
        indices = torch.LongTensor([[[triplet[0], triplet[2]] for triplet in batch_of_triplets] for batch_of_triplets in support_set])
        batch_nodes = np.array(list(set(indices.flatten().tolist())))

        embeddings, node_ids = self.rgcn(batch_nodes, background_graph)

        m = dict()
        for i, n in enumerate(node_ids):
            m[n.item()] = i
        indices = torch.LongTensor([[[m[triplet[0]], m[triplet[2]]] for triplet in batch_of_triplets] for batch_of_triplets in support_set]).to(self.device)

        embeddings_mean = torch.mean(embeddings[indices], dim=(1, 2))  # [bs, shots, 2, embed_size] -> [bs, embed_size]
        out = self.mlp(embeddings_mean)
        out = torch.sigmoid(out)
        return out


# class RGCNModule(nn.Module):

#     def __init__(self, parameter, embeddings, dataset):
#         super(RGCNModule, self).__init__()
#         self.device = parameter['device']
#         self.background = dataset['background']
#         self.ent2id = dataset['ent2id']
#         self.rel2id = dataset['rel2id']
#         self.num_shots = parameter['few']
#         self.num_entities = parameter['num_entities']
#         self.embedding_dim = parameter['embed_dim']
#         self.out_dim = 1

#         self.embeddings = embeddings

#         self.conv1 = RGCNConv(
#             in_channels=self.embedding_dim,
#             out_channels=self.embedding_dim,
#             num_relations=parameter['num_relations'],
#             num_bases=parameter['num_bases'],
#         )
#         self.conv2 = RGCNConv(
#             in_channels=self.embedding_dim,
#             out_channels=self.embedding_dim,
#             num_relations=parameter['num_relations'],
#             num_bases=parameter['num_bases'],
#         )

#         self.mlp = nn.Sequential(
#             nn.Linear(in_features=self.embedding_dim * 2 * self.num_shots, out_features=self.embedding_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=self.embedding_dim, out_features=self.out_dim),
#         )

#     def forward(self, tasks, iseval):

#         support = tasks[0]
#         # format support and BG into RGCN format
#         edge_index, edge_type = self._collate(support)

#         if not iseval:
#             x = F.relu(
#                 self.conv1(self.embeddings.embedding, edge_index, edge_type,
#                            size=[self.num_entities, self.num_entities])
#             )

#             # embeddings here is the data, not the embeddings obj
#             self.final_embeddings = self.conv2(x, edge_index, edge_type, size=[
#                 self.num_entities, self.num_entities])  # out dim: [num_entities, embedding_dim]
#         else:
#             self.final_embeddings = self.final_embeddings.detach()

#         support_ids = lookup_ids(self.ent2id, data=tasks[0])
#         support = self.final_embeddings[support_ids]

#         flat_support = support.reshape(support.shape[0], -1)
#         out = self.mlp(flat_support)
#         out = torch.sigmoid(out)
#         return out

#     def _str_to_ids(self, batch):
#         batch_ids = torch.tensor(lookup_ids(self.ent2id, self.rel2id, batch)).to(self.device).transpose(0, -1)
#         batch_ids = batch_ids.reshape(3, -1)  # 3, num_shots * batch_size
#         edge_index = batch_ids[[0, 2]]
#         edge_type = batch_ids[1]

#         return edge_index, edge_type

#     def _collate(self, support_batch):
#         edge_index, edge_type = self._str_to_ids(support_batch)

#         # convert background graph to ids
#         background_ids = torch.tensor(lookup_ids(self.ent2id, self.rel2id, self.background)).transpose(0, 1).to(self.device)
#         background_edge_index = background_ids[[0, 2]]
#         background_edge_type = background_ids[1]

#         batch_size = len(support_batch)

#         input_edges = torch.zeros(2, batch_size * self.num_shots + background_edge_index.shape[1]).long().to(
#             device=self.device)
#         input_edges[:, :self.num_shots * batch_size] = edge_index
#         input_edges[:, self.num_shots * batch_size:] = background_edge_index

#         input_type = torch.zeros(batch_size * self.num_shots + background_edge_type.shape[0]).long().to(
#             device=self.device)
#         input_type[:self.num_shots * batch_size] = edge_type
#         input_type[self.num_shots * batch_size:] = background_edge_type

#         return input_edges, input_type
