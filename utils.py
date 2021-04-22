import os
import json
import logging


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def initialize_experiment(params):
    params['ckpt_dir'] = os.path.join(params['state_dir'], params['experiment_name'], 'checkpoint')
    if not os.path.isdir(params['ckpt_dir']):
        print("creating new ckpt_dir",params['ckpt_dir'])
        os.makedirs(params['ckpt_dir'])


    pre = 'synthetic_' if params['is_synthetic'] else ''
    params['log_dir'] = os.path.join(params['log_dir'], pre + params['experiment_name'])
    if not os.path.isdir(params['log_dir']):
        os.makedirs(params['log_dir'])
        print("creating new log_dir",params['log_dir'])

    params['state_dir'] = os.path.join(params['state_dir'], params['experiment_name'])
    if not os.path.isdir(params['state_dir']):
        print("creating new state_dir",params['state_dir'])
        os.makedirs(params['state_dir'])

    # logging
    with open(os.path.join(params['log_dir'], "params.json"), 'w') as fout:
        json.dump(params, fout)

    file_handler = logging.FileHandler(os.path.join(params['log_dir'], 'res.log'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                           in sorted(params.items())))
    logging.info('============================================')
