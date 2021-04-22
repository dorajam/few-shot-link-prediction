import copy
import json
import random

import dgl
import numpy as np
import pandas as pd
import torch
from networkx.algorithms.components import connected_components
import community as community_louvain


def read_dataset(dataset, data_mode, add_inverse_edges):
    data_path = "./data/" + dataset

    data_dir = {
        'train_tasks_in_train': '/train_tasks_in_train.json',
        'train_tasks': '/train_tasks.json',
        'test_tasks': '/test_tasks.json',
        'dev_tasks': '/dev_tasks.json',

        'rel2candidates_in_train': '/rel2candidates_in_train.json',
        'rel2candidates': '/rel2candidates.json',

        'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
        'e1rel_e2': '/e1rel_e2.json',

        'ent2ids': '/ent2ids',
        'ent2vec': '/ent2vec.npy',
    }

    for k, v in data_dir.items():
        data_dir[k] = data_path + v

    tail = ''
    if data_mode == 'In-Train':
        tail = '_in_train'

    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks' + tail]))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))
    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates' + tail]))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2' + tail]))

    # load background graph
    print("preparing background graph {} ... ...".format(tail))
    whole_graph = json.load(open(data_dir['train_tasks_in_train']))

    if data_mode == 'Pre-Train':
        print('loading embedding ... ...')
        dataset['ent2emb'] = np.load(data_dir['ent2vec'])

    # additional params
    all_tasks = copy.deepcopy(whole_graph)
    all_tasks.update(dataset['dev_tasks'])
    all_tasks.update(dataset['test_tasks'])
    rel2id = get_relation_ids(all_tasks, data_path)

    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
    dataset['rel2id'] = rel2id

    dataset['background'] = build_background_graph(whole_graph, dataset['ent2id'], dataset['rel2id'], add_transpose=add_inverse_edges)

    return dataset


def get_relation_ids(all_tasks, data_path):
    all_relations = all_tasks.keys()

    rel2idx = {rel: idx for idx, rel in enumerate(all_relations)}
    idx2rel = {idx: rel for idx, rel in enumerate(all_relations)}

    json.dump(rel2idx, open(data_path + '/rel2idx.json', 'w'))
    json.dump(idx2rel, open(data_path + '/idx2rel.json', 'w'))

    return rel2idx


def build_background_graph(whole_graph, ent2id, rel2id, add_transpose=False):

    background_graph = dgl.DGLGraph(multigraph=True)

    background_graph.add_nodes(len(ent2id.keys()))

    rels = set(whole_graph.keys())
    for rel in rels:
        background_triples = whole_graph[rel]
        background_triple_ids = np.array([[ent2id[h], rel2id[r], ent2id[t]] for (h, r, t) in background_triples])
        background_graph.add_edges(background_triple_ids[:, 0], background_triple_ids[:, 2], {'type': torch.tensor(background_triple_ids[:, 1])})
        if add_transpose:
            background_graph.add_edges(background_triple_ids[:, 2], background_triple_ids[:, 0], {'type': torch.tensor(len(rels) + background_triple_ids[:, 1])})

    print('Background graph created for {} relations.'.format(max(background_graph.edata['type']) + 1))
    return background_graph


class SyntheticDataLoader(object):

    def __init__(self, dataset, parameter, step='train'):
        self.ent2id = dataset['ent2id']
        self.id2ent = {str(idx): ent for ent, idx in self.ent2id.items()}

        self.communities, self.entities = self.get_communities(dataset)
        self.num_entities = len(self.entities)
        self.curr_rel_idx = 0

        original_test_tasks = dataset[step + '_tasks']

        print('Generating synthetic relations ...')

        # symmetric
        symmetric_rels = ['symmetric_rel_' + str(idx) for idx in range(len(original_test_tasks))]
        symmetric_tasks = self.generate_symmetric_triples(symmetric_rels, original_test_tasks)

        # transitive
        transitive_rels = ['transitive_rel_' + str(idx) for idx in range(len(original_test_tasks))]
        transitive_tasks = self.generate_transitive_triples(transitive_rels, original_test_tasks)

        # positional
        positional_rels = ['positional_rel_' + str(idx) for idx in range(len(original_test_tasks))]
        positional_tasks = self.generate_positional_triples(positional_rels, original_test_tasks, dataset)

        # merge tasks
        self.tasks = symmetric_tasks
        self.tasks.update(transitive_tasks)
        self.tasks.update(positional_tasks)

        self.all_rels = self.tasks.keys()
        self.num_rels = len(self.tasks)
        print('Finished generating {} synthetic relations'.format(self.num_rels))

        self.rel2id = {rel: -idx for idx, rel in enumerate(self.all_rels)}  ## TODO: confirm these negative ids are not problematic
        self.id2rel= {str(idx): rel for rel, idx in self.rel2id.items()}  ## TODO: confirm these negative ids are not problematic

        # build dict for synthetic "observed" triples
        all_triples = [tri for task in self.tasks.values() for tri in task]
        self.e1rel_e2 = {e1+rel: e2 for [e1,rel,e2] in all_triples}

        # new uniformly sampled candidate tails
        original_rel2candidates = dataset['rel2candidates']
        original_test_rel2candidates = [original_rel2candidates[rel] for rel in original_test_tasks.keys()]
        self.rel2candidates = self.generate_synthetic_candidates(self.all_rels, original_test_rel2candidates)

        # parameters
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']
        self.seed = parameter['seed']

        # for each rel: use test triples[k_shot:] as query, and test triples[:k_shot] as support
        if step != 'train':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

    def get_communities(self, dataset):
        train_tasks = dataset['train_tasks']
        rel2id = dataset['rel2id']  # for non-synthetic rels
        nell_graph = build_background_graph(train_tasks, self.ent2id, rel2id)

        print('Converting to a networkx graph ....')
        nx_nell_g = nell_graph.to_networkx().to_undirected()

        print('Generating communities ....')
        components = list(connected_components(nx_nell_g))
        largest_component = sorted(components, key=lambda r: -len(r))[0]

        # Operating on subgraph of largest component
        new_graph = nx_nell_g.subgraph(largest_component)
        new_nodes = new_graph.nodes
        print('Taking largest component of size: ',len(new_nodes))
        communities = community_louvain.best_partition(new_graph, random_state=42)
        community_df = [(node, community_id) for node, community_id in communities.items()]
        community_df = pd.DataFrame(community_df, columns=['node', 'community_id'])

        print('Generating {} number of communities...'.format(community_df['community_id'].max() + 1))

        community_sizes = community_df.groupby('community_id').agg(count=('node', 'count'))
        print('Mean community size: ', community_sizes['count'].mean())
        print('Median community size: ', community_sizes['count'].median())
        print('Min community size: ', community_sizes['count'].min())
        print('Max community size: ', community_sizes['count'].max())

        node_clusters = community_df.groupby('community_id').apply(lambda c: list(c['node'])).values.tolist()
        assert len(new_nodes) == len([e for ls in node_clusters for e in ls]), 'Clusters should only contain nodes in the largest component.'
        return node_clusters, np.array(new_nodes)

    def generate_symmetric_triples(self, relations, original_test_tasks):
        # only sample half of the required entities, as we add its symmetric counterpart
        num_triples_to_sample = [r for r in map(lambda r: len(r) // 2 , original_test_tasks.values())]

        tasks = {rel: [] for rel in relations}

        for idx, rel in enumerate(relations):
            num_triples = num_triples_to_sample[idx]

            for _ in range(num_triples):
                # sample H,T for synthetic triples
                while True:
                    ids = np.random.randint(0, self.num_entities, size=2)  # not without replacement
                    if len(set(ids)) == 2:
                        break
                entities = self.entities[ids]
                # entities = np.random.choice(self.entities, size=2, replace=False)  # too slow

                # adds symmetric triples
                tasks[rel].append([self.id2ent[str(entities[0])], rel, self.id2ent[str(entities[1])]])
                tasks[rel].append([self.id2ent[str(entities[1])], rel, self.id2ent[str(entities[0])]])

        return tasks

    def generate_positional_triples(self, relations, original_test_tasks, dataset):
        num_triples_to_sample = [r for r in map(lambda r: len(r) , original_test_tasks.values())]

        tasks = {rel: [] for rel in relations}

        for idx, rel in enumerate(relations):
            num_triples = num_triples_to_sample[idx]

            # sample a community
            sampled_communities = np.random.choice(self.communities, size=num_triples, replace=True)

            # sample h,t
            for c_i in sampled_communities:
                while True:
                    ids = np.random.randint(0, len(c_i), size=2)  # not without replacement
                    if len(set(ids)) == 2:
                        break
                entities = np.array(c_i)[ids]
                # entities = np.random.choice(c_i, size=2, replace=False)

                # add triple
                tasks[rel].append([self.id2ent[str(entities[0])], rel, self.id2ent[str(entities[1])]])

        return tasks


    def generate_transitive_triples(self, relations, original_test_tasks):
        # only sample 1/3 of the required entities, as we add its transitive counterparts
        num_triples_to_sample = [r for r in map(lambda r: len(r) // 3 , original_test_tasks.values())]

        tasks = {rel: [] for rel in relations}

        for idx, rel in enumerate(relations):
            num_triples = num_triples_to_sample[idx]
            for _ in range(num_triples):
                # sample h,t
                while True:
                    ids = np.random.randint(0, self.num_entities, size=3)  # not without replacement
                    if len(set(ids)) == 3:
                        break
                entities = self.entities[ids]
                # entities = np.random.choice(self.entities, size=3, replace=False)

                # adds transitive triples
                if len(entities) == 3:
                    tasks[rel].append([self.id2ent[str(entities[0])], rel, self.id2ent[str(entities[1])]])
                    tasks[rel].append([self.id2ent[str(entities[1])], rel, self.id2ent[str(entities[2])]])
                    tasks[rel].append([self.id2ent[str(entities[0])], rel, self.id2ent[str(entities[2])]])

        return tasks

    def generate_synthetic_candidates(self, relations, original_rel2candidates):
        # tile samples to generate for each synthetic type
        num_candidates_to_sample = np.tile([len(cands) for cands in original_rel2candidates], 3)

        rel2candidates = {}
        for idx, rel in enumerate(relations):
            num_cands = num_candidates_to_sample[idx]

            curr = num_cands
            final_ids = np.array([])
            while True:
                ids = np.random.randint(0, self.num_entities, size=curr)  # not without replacement
                curr = curr - len(set(ids))
                final_ids = np.append(final_ids, ids)
                if curr == 0:
                    break
            sampled_entities = self.entities[final_ids.astype(np.int)]
            # sampled_entities = np.random.choice(self.entities, size=num_cands, replace=False)
            rel2candidates[rel] = [self.id2ent[str(ent)] for ent in sampled_entities]

        return rel2candidates

    def get_id(self, triplets):
        return [[self.ent2id[h], self.rel2id[r], self.ent2id[t]] for (h, r, t) in triplets]

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            print('Finished evaluating {} queries.'.format(self.num_tris))
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])

        return [[self.get_id(support_triples)], [self.get_id(support_negative_triples)], [self.get_id([query_triple])], [self.get_id(negative_triples)]], self.rel2id[curr_rel]



class DataLoader(object):

    def __init__(self, dataset, parameter, step='train'):
        self.ent2id = dataset['ent2id']
        self.rel2id = dataset['rel2id']
        self.curr_rel_idx = 0
        self.tasks = dataset[step + '_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']

        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']
        self.seed = parameter['seed']

        if step != 'train':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

    def get_id(self, triplets):
        return [[self.ent2id[h], self.rel2id[r], self.ent2id[t]] for (h, r, t) in triplets]

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx]
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels  # shift current relation idx to next
        curr_cand = self.rel2candidates[curr_rel]
        while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[curr_rel]

        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few + self.nq)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]

        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
            support_negative_triples.append([e1, rel, negative])

        negative_triples = []
        for triple in query_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
            negative_triples.append([e1, rel, negative])

        return self.get_id(support_triples), self.get_id(support_negative_triples), self.get_id(query_triples), self.get_id(negative_triples), self.rel2id[curr_rel]

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)

        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])

        return [[self.get_id(support_triples)], [self.get_id(support_negative_triples)], [self.get_id([query_triple])], [self.get_id(negative_triples)]], self.rel2id[curr_rel]

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few:]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # get current triple
        query_triple = self.tasks[curr_rel][self.few:][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])

        return [[self.get_id(support_triples)], [self.get_id(support_negative_triples)], [self.get_id([query_triple])], [self.get_id(negative_triples)]], self.rel2id[curr_rel]
