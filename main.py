import argparse
import logging
import random

import numpy as np
import torch
import wandb  # dashboard

from data_loader import read_dataset, DataLoader, SyntheticDataLoader
from trainer import Trainer
from models.meta_learner import MetaLearner
from utils import initialize_experiment


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = argparse.ArgumentParser()

    # Experiment setup params
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, action='store_true')
    args.add_argument("-exp", "--experiment_name", default="deault_name", type=str)
    args.add_argument("-is_synthetic", "--is_synthetic", default=False, action='store_true')
    args.add_argument("-dash", "--dashboard", default=False, action='store_true')
    args.add_argument("-seed", "--seed", default=42, type=int)
    args.add_argument("-abla", "--ablation", default=False, action='store_true')
    args.add_argument("-gpu", "--device", default=0, type=int)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-save_all", "--save_all", default=False, action='store_true')

    # Data processing pipeline params
    args.add_argument("-data", "--dataset", default="Wiki-One", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-form", "--data_form", default="In-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-few", "--few", default=5, type=int)
    args.add_argument("-nq", "--num_query", default=10, type=int)
    args.add_argument("-inverse", "--add_inverse_edges", default=False, action='store_true')

    # Training regime params
    args.add_argument("-iter", "--iterations", default=100000, type=int)
    args.add_argument("-prt_iter", "--print_iter", default=100, type=int)
    args.add_argument("-eval_iter", "--eval_iter", default=1000, type=int)
    args.add_argument("-ckpt_iter", "--checkpoint_iter", default=1000, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-es", "--early_stopping_patience", default=30, type=int)
    args.add_argument("-metric", "--metric", default="Hits@10", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])
    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-reg", "--reg_weight", default=0.01, type=float)
    args.add_argument("-lr_step", "--lr_step", default=10, type=float)
    args.add_argument("-lr_rate", "--lr_rate", default=1., type=float)

    # Model params
    args.add_argument("-rmeta", "--rmeta_learner", default="MetaR", choices=["MetaR", "Modular", "Simple"])
    args.add_argument("-module", "--module_type", default="RGCN", choices=["MLP", "RGCN"])
    args.add_argument('-rgcn', '--use_rgcn', action='store_true', default=False)

    # rgcn
    args.add_argument("-nb", "--num_bases", default=4, type=int)
    args.add_argument("-emb", "--embed_dim", default=50, type=int)
    args.add_argument("-rgcn_emb", "--rgcn_embed_dim", default=20, type=int)
    args.add_argument("-gcn_l", "--num_gcn_layers", default=2, type=int)
    args.add_argument("-e_dp", "--edge_dropout", default=0.0, type=float)
    args.add_argument("-ns", "--neighborhood_sample_rate", default=20, type=int)
    args.add_argument('--has_attn', '-attn', action='store_true', default=False)
    args.add_argument("-agg", "--gnn_agg_type", default="sum", choices=["sum", "mlp", "gru"])
    # prototypes
    args.add_argument("-pd", "--prototype_dim", default=100, type=int)
    args.add_argument("-np", "--num_prototypes", default=4, type=int)
    args.add_argument("-dp", "--dropout", default=0.5, type=float)

    args = args.parse_args()

    params = {}
    for k, v in vars(args).items():
        params[k] = v

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    dataset = read_dataset(params['dataset'], params['data_form'], add_inverse_edges=params['add_inverse_edges'])

    params['num_entities'] = len(dataset['ent2id'].keys())
    params['num_relations'] = max(dataset['background'].edata['type']).item() + 1

    initialize_experiment(params)
    if params['dashboard']:
        wandb.init(project="logic-nets", config=params, name=args.experiment_name)

    if params['device'] < 0:
        params['device'] = torch.device('cpu')
    else:
        params['device'] = torch.device('cuda:' + str(params['device']))

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    if params['is_synthetic']:
        synthetic_test_data_loader = SyntheticDataLoader(dataset, params, step='test')
        data_loaders = [train_data_loader, dev_data_loader, test_data_loader, synthetic_test_data_loader]
    else:
        data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # model
    meta_learner = MetaLearner(params, background_graph=dataset['background'])

    model_params = list(meta_learner.parameters())
    logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))
    if params['dashboard']:
        wandb.watch(meta_learner, log="all")

    # trainer
    trainer = Trainer(meta_learner, data_loaders, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['experiment_name'])
        trainer.reload()
        trainer.eval(isTest=True, save_all=params['save_all'])
    elif params['step'] == 'test':
        print(params['experiment_name'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(isTest=True)
        else:
            trainer.eval(isTest=True, save_all=params['save_all'], synthetic=params['is_synthetic'])
    elif params['step'] == 'dev':
        print(params['experiment_name'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(isTest=False)
        else:
            trainer.eval(isTest=False)
