import os
import sys
import shutil
import logging
import time

import torch
import wandb
import numpy as np

from json import load
from collections import Counter
from tensorboardX import SummaryWriter
from utils import get_lr


class Trainer:
	def __init__(self, model, data_loaders, parameter):
		self.parameter = parameter
		# dir
		self.state_dir = parameter['state_dir']
		self.ckpt_dir = parameter['state_dir']
		self.log_dir = parameter['log_dir']
		self.state_dict_file = ''
		self.idx2rel = load(open('data/' + self.parameter['dataset'] + '/idx2rel.json', 'r'))
		self.ent2ids = load(open('data/' + self.parameter['dataset'] + '/ent2ids', 'r'))
		self.id2ent = {str(idx): ent for ent, idx in self.ent2ids.items()}

		# data loader
		self.train_data_loader = data_loaders[0]
		self.dev_data_loader = data_loaders[1]
		self.test_data_loader = data_loaders[2]
		if parameter['is_synthetic']:
			self.synthetic_test_data_loader = data_loaders[3]

		# triples
		triplets = list(self.train_data_loader.tasks.values())
		all_entities = np.array([[self.ent2ids[t[0]], self.ent2ids[t[2]]] for rel in triplets for t in rel]).flatten()
		self.c = Counter(all_entities)

		# parameters
		self.few = parameter['few']
		self.num_query = parameter['num_query']
		self.batch_size = parameter['batch_size']
		self.learning_rate = parameter['learning_rate']
		self.early_stopping_patience = parameter['early_stopping_patience']

		# epoch
		self.iterations = parameter['iterations']
		self.print_iter = parameter['print_iter']
		self.eval_iter = parameter['eval_iter']
		self.checkpoint_iter = parameter['checkpoint_iter']

		self.device = parameter['device']

		# tensorboard log writer
		if parameter['step'] == 'train':
			self.writer = SummaryWriter(os.path.join(parameter['log_dir']))

		# model
		self.meta_learner = model
		self.meta_learner.to(self.device)

		# optimizer
		self.optimizer = torch.optim.Adam(self.meta_learner.parameters(), self.learning_rate)

		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.parameter['iterations'] / parameter['lr_step'], gamma=parameter['lr_rate'])

		# load state_dict and params
		if parameter['step'] in ['test', 'dev']:
			self.reload()

	def reload(self):
		if self.parameter['eval_ckpt'] is not None:
			state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
		else:
			state_dict_file = os.path.join(self.state_dir, 'state_dict')
		self.state_dict_file = state_dict_file
		logging.info('Reload state_dict from {}'.format(state_dict_file))
		print('reload state_dict from {}'.format(state_dict_file))
		state = torch.load(state_dict_file, map_location=self.device)
		if os.path.isfile(state_dict_file):
			self.meta_learner.load_state_dict(state)
			self.meta_learner.fill_bg_with_data()
		else:
			raise RuntimeError('No state dict in {}!'.format(state_dict_file))

	def save_checkpoint(self, iteration):
		torch.save(self.meta_learner.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(iteration) + '.ckpt'))

	def del_checkpoint(self, iteration):
		path = os.path.join(self.ckpt_dir, 'state_dict_' + str(iteration) + '.ckpt')
		if os.path.exists(path):
			os.remove(path)
		else:
			raise RuntimeError('No such checkpoint to delete: {}'.format(path))

	def save_best_state_dict(self, best_epoch):
		shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
					os.path.join(self.state_dir, 'state_dict'))

	def write_training_log(self, data, iteration):
		self.writer.add_scalar('Training_Loss', data['Loss'], iteration)
		if self.parameter['dashboard']:
			wandb.log({'train_loss': data['Loss']})

	def write_validating_log(self, data, iteration):
		self.writer.add_scalar('Validating_MRR', data['MRR'], iteration)
		self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], iteration)
		self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], iteration)
		self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], iteration)
		if self.parameter['dashboard']:
			wandb.log({
				'val_mrr': data['MRR'],
				'val_hits10': data['Hits@10'],
				'val_hits5': data['Hits@5'],
				'val_hits1': data['Hits@1']})

	def logging_training_data(self, data, iteration):
		logging.info("Iter: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
			iteration, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

	def logging_eval_data(self, data, state_path, isTest=False):
		setname = 'dev set'
		if isTest:
			setname = 'test set'
		logging.info("Eval {} on {}".format(state_path, setname))
		logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
			data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

	def rank_predict(self, data, x, ranks, tmp=None):
		# query_idx is the idx of positive score
		query_idx = x.shape[0] - 1
		# sort all scores with descending, because more plausible triple has higher score
		_, idx = torch.sort(x, descending=True)
		rank = list(idx.cpu().numpy()).index(query_idx) + 1
		ranks.append(rank)
		h10, h5, h1 = 0., 0., 0.
		# update data
		if rank <= 10:
			data['Hits@10'] += 1.
			h10 = 1
		if rank <= 5:
			data['Hits@5'] += 1.
			h5 = 1
		if rank == 1:
			data['Hits@1'] += 1.
			h1 = 1
		data['MRR'] += 1.0 / rank
		mrr = 1. / rank
		if tmp:
			tmp['mrr'] = mrr
			tmp['H10'] =  h10
			tmp['H5'] = h5
			tmp['H1'] = h1

	def do_one_step(self, task, curr_rel=''):
		loss, p_score, n_score = 0, 0, 0
		if self.meta_learner.training:
			self.optimizer.zero_grad()
			p_score, n_score = self.meta_learner(task, curr_rel)
			y = torch.Tensor([1]).to(self.device)

			loss = self.meta_learner.loss_func(p_score, n_score, y)

			if self.parameter['rmeta_learner'] == 'Modular':
				prototypes = [p for p in self.meta_learner.relation_meta_learner.prototypes.parameters()][0]
				ortho_regularizer = torch.norm(
					torch.matmul(prototypes, prototypes.transpose(0, 1)) - torch.eye(self.parameter['num_prototypes']).to(self.device)
				)
				loss += self.parameter['reg_weight'] * ortho_regularizer
			loss.backward()
			self.optimizer.step()

		elif curr_rel != '':
			p_score, n_score = self.meta_learner(task, curr_rel)
			y = torch.Tensor([1]).to(self.device)
			loss = self.meta_learner.loss_func(p_score, n_score, y)
		return loss, p_score, n_score

	def train(self):
		# initialization
		best_iter = 0
		best_value = 0
		bad_counts = 0
		tic = time.time()
		# training by iter
		for e in range(self.iterations):
			self.meta_learner.train()
			# sample one batch from data_loader
			train_task, curr_rel = self.train_data_loader.next_batch()
			loss, _, _ = self.do_one_step(train_task, curr_rel=curr_rel)
			# print the loss on specific iter
			if e % self.print_iter == 0:
				loss_num = loss.item()
				self.write_training_log({'Loss': loss_num}, e)
				logging.info("Iter: {}\tLoss: {:.4f}\tTime: {:.4f}\tlr: {:.4f}".format(e, loss_num, time.time() - tic, get_lr(self.optimizer)))
				tic = time.time()
			# save checkpoint on specific iter
			if e % self.checkpoint_iter == 0 and e != 0:
				logging.info('Iter  {} has finished, saving...'.format(e))
				self.save_checkpoint(e)
			# do evaluation on specific iter
			if e % self.eval_iter == 0 and e != 0:
				logging.info('Iter  {} has finished, validating...'.format(e))

				valid_data = self.eval(isTest=False)
				self.write_validating_log(valid_data, e)

				metric = self.parameter['metric']
				# early stopping checking
				if valid_data[metric] > best_value:
					best_value = valid_data[metric]
					best_iter = e
					logging.info('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
					bad_counts = 0
					# save current best
					self.save_checkpoint(best_iter)
					self.save_best_state_dict(best_iter)
				else:
					logging.info('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
						metric, best_value, best_iter, bad_counts))
					bad_counts += 1

				if bad_counts >= self.early_stopping_patience:
					logging.info('\tEarly stopping at iteration %d' % e)
					break
			self.scheduler.step()

		logging.info('Training has finished')
		logging.info('\tBest iteration is {0} | {1} of valid set is {2:.3f}'.format(best_iter, metric, best_value))
		self.save_best_state_dict(best_iter)
		logging.info('Finish')

	def _diagnostics(self, query_head, correct_tail, rel_id, support, candidates):
		"""
		For given eval task, take the query head and current relation
		and return: the number of candidates for the relation and
		the number of occurances of the head entity in the training data.
		"""

		num_of_candidates = len(candidates)

		head_occurance = self.c[query_head]
		tail_occurance = self.c[correct_tail]

		tail_in_support = 0
		head_in_support = 0

		support_entities = np.array([[triple[0], triple[2]] for triple in support]).flatten()
		support_occurances = 0
		for e in support_entities:
			support_occurances += self.c[e]
			if e == query_head:
				head_in_support += 1
			if e == correct_tail:
				tail_in_support += 1

		return num_of_candidates, head_occurance, support_occurances, tail_occurance, head_in_support, tail_in_support, support_entities

	def eval(self, isTest=False, save_all=False, synthetic=False):
		self.meta_learner.eval()
		# clear sharing rel_q
		self.meta_learner.rel_q_sharing = dict()

		if not synthetic:
			if isTest:
				data_loader = self.test_data_loader
			else:
				data_loader = self.dev_data_loader
		else:
			print('Using synthetic dataloader...')
			data_loader = self.synthetic_test_data_loader

		data_loader.curr_tri_idx = 0

		# initial return data of validation
		data = {'loss': 0, 'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
		tmp = {'mrr': 0, 'H10': 0, 'H5': 0, 'H1': 0}
		ranks = []

		t = 0
		temp = dict()
		tic = time.time()
		diagnostics = []

		while True:
			# sample all the eval tasks
			eval_task, curr_rel = data_loader.next_one_on_eval()
			# at the end of sample tasks, a symbol 'EOT' will return
			if eval_task == 'EOT':
				break
			t += 1
			# pdb.set_trace()
			l, p_score, n_score = self.do_one_step(eval_task, curr_rel=curr_rel)

			x = torch.cat([n_score, p_score], 1).squeeze()

			self.rank_predict(data, x, ranks, tmp)

			if save_all:
				if synthetic:
					relation = data_loader.id2rel[str(curr_rel)]
				else:
					relation = self.idx2rel[str(curr_rel)]
				candidates = data_loader.rel2candidates[relation]

				query_head = eval_task[2][0][0][0]
				correct_tail = eval_task[2][0][0][2]
				support = eval_task[0][0]

				candidate_size, query_head_seen, support_entities_seen, correct_tail_seen, head_in_support, tail_in_support, support_entities = self._diagnostics(
					query_head, correct_tail, curr_rel, support, candidates)

				diagnostics.append({
					'MRR': tmp['mrr'],
					'H10': tmp['H10'],
					'H5': tmp['H5'],
					'H1': tmp['H1'],
					'query_head': self.id2ent[str(query_head)],
					'correct_tail': self.id2ent[str(correct_tail)],
					'query_head_seen': query_head_seen,
					'correct_tail_seen': correct_tail_seen,
					'head_seen_in_support': head_in_support,
					'tail_seen_in_support': tail_in_support,
                    'support_entities': support_entities,
					'support_entities_seen': support_entities_seen,
					'candidate_size': candidate_size,
					'rel': relation,
                    'synthetic_type': relation[:3] if self.parameter['is_synthetic'] else 'not_synthetic',
					'data': self.parameter['dataset'],
					'num_shots': self.parameter['few'],
				})

			data['loss'] += l.item()
			# print current temp data dynamically
			for k in data.keys():
				temp[k] = data[k] / t
			sys.stdout.write("{}\tVal. loss: {:.3f}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
				t, temp['loss'], temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
			sys.stdout.flush()

		flag = '_synthetic' if self.parameter['is_synthetic'] else ''
		file = self.parameter['experiment_name'] + flag + '_diagnostics.json'
		print('Saving raw diagnostics under: {}'.format(file))
		torch.save(diagnostics, os.path.join('best_models', 'diagnostics', file))

		toc = time.time()
		# print overall evaluation result and return it
		for k in data.keys():
			data[k] = round(data[k] / t, 3)

		logging.info("{}  Val. loss: {:.3f}  MRR: {:.3f}  Hits@10: {:.3f}  Hits@5: {:.3f}  Hits@1: {:.3f}  Time: {:.3f}\r".format(
			t, data['loss'], data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1'], toc - tic))

		if isTest:
			with open(os.path.join(self.parameter['log_dir'], flag + 'test_scores.txt'), "w") as f:
				f.write('MRR | Hits@10 | Hits@5 | Hits@1 : {:.4f} | {:.4f} | {:.4f} | {:.3f}\n'.format(data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

		return data

	def eval_by_relation(self, isTest=False, iteration=None):
		self.meta_learner.eval()
		self.meta_learner.rel_q_sharing = dict()

		if isTest:
			data_loader = self.test_data_loader
		else:
			data_loader = self.dev_data_loader
		data_loader.curr_tri_idx = 0

		all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
		all_t = 0
		all_ranks = []

		for rel in data_loader.all_rels:
			print("rel: {}, num_cands: {}, num_tasks:{}".format(
				rel, len(data_loader.rel2candidates[rel]), len(data_loader.tasks[rel][self.few:])))
			data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
			temp = dict()
			t = 0
			ranks = []
			while True:
				eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
				if eval_task == 'EOT':
					break
				t += 1

				_, p_score, n_score = self.do_one_step(eval_task, curr_rel=rel)
				x = torch.cat([n_score, p_score], 1).squeeze()

				self.rank_predict(data, x, ranks)

				for k in data.keys():
					temp[k] = data[k] / t
				sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
					t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
				sys.stdout.flush()

			print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
				t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

			for k in data.keys():
				all_data[k] += data[k]
			all_t += t
			all_ranks.extend(ranks)

		print('Overall')
		for k in all_data.keys():
			all_data[k] = round(all_data[k] / all_t, 3)
		print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
			all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@1']))

		return all_data
