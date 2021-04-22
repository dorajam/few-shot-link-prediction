import functools
import numpy as np
import torch
from torch import nn
from .relation_meta_learner import MetaR, ModularPrototypes, SimplePrototype
from .rgcn_model import RGCN


class MetaLearner(nn.Module):

    def __init__(self, parameter, background_graph=None):
        super(MetaLearner, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.use_rgcn = parameter['use_rgcn']
        self.step = parameter['step']

        self.embeddings = nn.Parameter(torch.FloatTensor(parameter['num_entities'], parameter['embed_dim']))
        nn.init.xavier_uniform_(self.embeddings.data)

        self.final_embeddings = nn.Parameter(torch.FloatTensor(parameter['num_entities'], parameter['embed_dim']))
        nn.init.xavier_uniform_(self.final_embeddings.data)

        self.background_graph = background_graph
        self.background_graph.edata['type'] = self.background_graph.edata['type'].to(self.device)
        self.background_graph.ndata['feat'] = self.embeddings
        self.background_graph.ndata['repr'] = self.embeddings.detach().to(self.device)

        if self.use_rgcn:
            self.rgcn = RGCN(parameter, is_module=False)

        if parameter['rmeta_learner'] == 'MetaR':
            self.relation_meta_learner = MetaR(parameter=parameter)

        if parameter['rmeta_learner'] == 'Modular':
            self.relation_meta_learner = ModularPrototypes(parameter=parameter)

        if parameter['rmeta_learner'] == 'Simple':
            self.relation_meta_learner = SimplePrototype(parameter=parameter)

        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def fill_bg_with_data(self):
        self.background_graph.ndata['feat'] = self.embeddings
        self.background_graph.ndata['repr'] = self.final_embeddings.detach().to(self.device)

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def get_embeddings(self, tasks):
        indices = [torch.LongTensor([[[triplet[0], triplet[2]] for triplet in batch_of_triplets] for batch_of_triplets in t]).to(self.device) for t in tasks]
        if self.use_rgcn:
            if self.training:
                batch_nodes = np.array(functools.reduce(lambda x, y: list(set(x + y)), [list(ind.flatten().cpu().numpy()) for ind in indices]))
                embeddings, nid = self.rgcn(batch_nodes, self.background_graph)
                m = dict()
                for i, n in enumerate(nid):
                    m[n.item()] = i
                indices = [torch.LongTensor([[[m[triplet[0]], m[triplet[2]]] for triplet in batch_of_triplets] for batch_of_triplets in t]).to(self.device) for t in tasks]
            else:
                # embeddings = self.background_graph.ndata['repr'].detach()
                self.final_embeddings.data = self.background_graph.ndata['repr'].detach()
                embeddings = self.final_embeddings
        else:
            embeddings = self.embeddings
        support, support_negative, query, query_negative = [embeddings[ids] for ids in indices]

        return support, support_negative, query, query_negative

    def forward(self, tasks, curr_rel=''):
        support, support_negative, query, negative = self.get_embeddings(tasks)

        num_shots = support.shape[1]  # num of few
        num_support_negatives = support_negative.shape[1]  # num of support negative
        num_queries = query.shape[1]  # num of query
        num_query_negatives = negative.shape[1]  # num of query negative

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if not self.training and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            r_meta = self.relation_meta_learner(tasks[0], support, background_graph=self.background_graph)
            r_meta.retain_grad()
            # relation for support
            rel_s = r_meta.expand(-1, num_shots + num_support_negatives, -1, -1)

            # split on e1/e2 and concat on pos/neg
            sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, num_shots)

            y = torch.Tensor([1]).to(self.device)
            self.zero_grad()
            loss = self.loss_func(p_score, n_score, y)
            loss.backward(retain_graph=self.training)

            if not self.abla:
                grad_meta = r_meta.grad
                rel_q = r_meta - self.beta * grad_meta
            else:
                rel_q = r_meta

            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_queries + num_query_negatives, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_queries)

        return p_score, n_score


class EmbeddingLearner(nn.Module):

    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score
