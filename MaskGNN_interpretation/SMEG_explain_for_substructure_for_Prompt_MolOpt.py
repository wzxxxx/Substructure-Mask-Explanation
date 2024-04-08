import numpy as np
import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
import pickle as pkl
import argparse

# fix parameters of model
def SMEG_explain_for_substructure(seed, data_name, model_name, batch_size=128, rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128,
               lr=0.0003, classification=True, sub_type='fg', group='train'):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    args['classification'] = classification
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = batch_size
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = rgcn_hidden_feats
    args['ffn_hidden_feats'] = ffn_hidden_feats
    args['rgcn_drop_out'] = 0
    args['ffn_drop_out'] = 0
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = model_name  # change
    args['data_name'] = data_name  # change
    args['bin_path'] = '../data/graph_data/{}_for_{}.bin'.format(args['data_name'], sub_type)
    args['group_path'] = '../data/graph_data/{}_group_for_{}.csv'.format(args['data_name'], sub_type)
    args['smask_path'] = '../data/graph_data/{}_smask_for_{}.npy'.format(args['data_name'], sub_type)
    args['seed'] = seed
    
    print('***************************************************************************************************')
    print('{}, seed {}, substructure type {}'.format(args['task_name'], args['seed']+1, sub_type))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        smask_path=args['smask_path'],
        random_shuffle=False
    )
    print("Molecule graph is loaded!")
    if group=='train':
        data_loader = DataLoader(dataset=train_set,
                                 batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)
    if group=='test':
        data_loader = DataLoader(dataset=test_set,
                                 batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)
    if args['classification']:
        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        loss_criterion = torch.nn.MSELoss(reduction='none')
    model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                 ffn_dropout=args['ffn_drop_out'],
                 rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                 rgcn_drop_out=args['rgcn_drop_out'],
                 classification=args['classification'])
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'] + '_' + str(seed + 1),
                            mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)
    pred_name = '{}_{}_{}_{}'.format(args['data_name'], model_name, sub_type, seed + 1)
    stop_test_list, _ = run_an_eval_epoch(args, model, data_loader, loss_criterion,
                                          out_path='../prediction/{}/{}_{}'.format(sub_type, pred_name, group))
    print('Mask prediction is generated!')


group = 'test'
parser = argparse.ArgumentParser(description='SME for substructure')
parser.add_argument('--data_name', type=str, help='the data name')
parser.add_argument('--model_name', type=str, help='the model name')
args = parser.parse_args()
data = args.data_name
model = args.model_name
for sub_type in ['fg', 'brics', 'murcko']:
    with open('../result/hyperparameter_{}.pkl'.format(model), 'rb') as f:
        hyperparameter = pkl.load(f)
    for i in range(10):
        SMEG_explain_for_substructure(seed=i, data_name=data, model_name=model, batch_size=8,
                                      rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                      ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                      lr=hyperparameter['lr'], classification=hyperparameter['classification'],
                                      sub_type=sub_type, group=group)







