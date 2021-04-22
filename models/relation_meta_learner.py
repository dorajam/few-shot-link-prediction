from collections import OrderedDict
import torch
from torch import nn
from .modules import MLPModule, RGCNModule


class SimplePrototype(nn.Module):

    def __init__(self, parameter):
        super(SimplePrototype, self).__init__()

        self.embedding_dim = parameter['embed_dim']
        self.device = parameter['device']
        self.prototypes = nn.Embedding(1, self.embedding_dim)
        self.prototype2out = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, support_set, support_emb, background_graph):
        batch_size = len(support_set)  # support batch size
        r_meta = self.prototype2out(self.prototypes.weight.expand(batch_size, -1))

        return r_meta.view(r_meta.shape[0], 1, 1, r_meta.shape[-1])


class ModularPrototypes(nn.Module):

    def __init__(self, parameter):
        super(ModularPrototypes, self).__init__()

        self.num_prototypes = parameter['num_prototypes']
        self.prototype_dim = parameter['prototype_dim']
        self.embedding_dim = parameter['embed_dim']
        self.num_entities = parameter['num_entities']
        self.out_dim = parameter['embed_dim']
        self.device = parameter['device']

        if parameter['module_type'] == 'MLP':
            module_class = MLPModule
        elif parameter['module_type'] == 'RGCN':
            module_class = RGCNModule

        module_list = []
        for idx in range(self.num_prototypes):
            module = module_class(parameter=parameter)
            module_list.append(module)

        self.module_list = nn.ModuleList(module_list)
        self.prototypes = nn.Embedding(self.num_prototypes, self.prototype_dim)
        self.prototype2out = nn.Linear(self.prototype_dim, self.out_dim)

    def forward(self, support_set, support_emb, background_graph):
        batch_size = len(support_set)  # support batch size
        prototypes = self.prototypes(torch.tensor(range(self.num_prototypes), device=self.device))
        attention_scores = torch.zeros(batch_size, self.num_prototypes, device=self.device)

        # Calculate prototype attention scores
        for idx in range(self.num_prototypes):
            # there are a set of embeddings per each model - ignore them, use initial embeddings in entity lookup
            attention_score = self.module_list[idx](support_set, support_emb, background_graph)
            attention_scores[:, idx] = attention_score.squeeze(1)

        # Take weighted sum over all prototypes, outputs r_meta.
        attended_prototypes = torch.matmul(attention_scores, prototypes)
        r_meta = self.prototype2out(attended_prototypes)

        return r_meta.view(r_meta.shape[0], 1, 1, r_meta.shape[-1])


class MetaR(nn.Module):

    def __init__(self, parameter):
        super(MetaR, self).__init__()
        self.embedding_dim = parameter['embed_dim']
        self.out_dim = parameter['embed_dim']
        self.device = parameter['device']

        if parameter['dataset'] == 'Wiki-One':
            num_hidden1 = 250
            num_hidden2 = 100

        elif parameter['dataset'] == 'NELL-One':
            num_hidden1 = 500
            num_hidden2 = 200

        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(2 * self.embedding_dim, num_hidden1)),
            ('bn', nn.BatchNorm1d(parameter['few'])),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=parameter['dropout'])),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, num_hidden2)),
            ('bn', nn.BatchNorm1d(parameter['few'])),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=parameter['dropout'])),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, self.out_dim)),
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

        return r_meta.view(r_meta.shape[0], 1, 1, r_meta.shape[-1])
