import numpy as np
from MaskGNN_interpretation import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from MaskGNN_interpretation.maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
import time


# fix parameters of model
def maskgnn_model_explain_for_no_mask(seed, task_name, rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128,
               lr=0.0003, classification=True, regularization_coefficients=0.2):
    args = {}
    args['device'] = "cpu"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    args['classification'] = classification
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 128
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = rgcn_hidden_feats
    args['ffn_hidden_feats'] = ffn_hidden_feats
    args['rgcn_drop_out'] = 0
    args['ffn_drop_out'] = 0
    args['regularization_coefficients'] = regularization_coefficients
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = task_name  # change
    args['bin_path'] = '../data/graph_data/' + args['data_name'] + '.bin'
    args['group_path'] = '../data/graph_data/' + args['data_name'] + '_group.csv'
    args['seed'] = seed
    
    print('***************************************************************************************************')
    print('{} regularization_coefficients {}, seed {}'.format(args['task_name'], args['regularization_coefficients'], args['seed']))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number, train_mean, train_var = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        classification=args['classification'],
        random_shuffle=False
    )
    args['train_mean'] = train_mean
    args['train_var'] = train_var
    print("Molecule graph is loaded!")
    train_loader = DataLoader(dataset=train_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    
    test_loader = DataLoader(dataset=test_set,
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
    stopper = EarlyStopping(patience=args['patience'], task_name=task_name + '_' + str(seed + 1),
                            mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)
    pred_name = 'prediction_{}_{}'.format(seed + 1, args['regularization_coefficients'])
    stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion,
                                          out_path='../prediction/' + args[
                                              'task_name'] + '_' + pred_name + '_train')
    stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion,
                                          out_path='../prediction/' + args[
                                              'task_name'] + '_' + pred_name + '_val')
    stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion,
                                          out_path='../prediction/' + args[
                                              'task_name'] + '_' + pred_name + '_test')
    print('Mask prediction is generated!')






# Mutagenicity
# task_name = 'Mutagenicity'
# hyperparameter = {'rgcn_hidden_feats':[[64, 64, 64], [128, 128, 128], [256, 256, 256], [128, 128], [64, 64, 64]], 'ffn_hidden_feats':[128, 128, 128, 128, 128],
#                 'rgcn_drop_out':[0, 0.1, 0.5, 0, 0.2], 'ffn_drop_out':[0.3, 0.2, 0.2, 0.2, 0.2], 'lr':[0.001, 0.0001, 0.001, 0.0001, 0.001],
#                 'classification':True}
# for i in range(5):
#     maskgnn_model_explain_for_no_mask(seed=i, task_name=task_name, rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'][i],
#                                      ffn_hidden_feats=hyperparameter['ffn_hidden_feats'][i],
#                                      lr=hyperparameter['lr'][i], classification=hyperparameter['classification'])


# task_name = 'ESOL'
# hyperparameter = {'rgcn_hidden_feats':[[256, 256], [64, 64, 64], [128, 128], [128, 128], [256, 256]],
#                   'ffn_hidden_feats':[128, 128, 128, 128, 128],
#                   'rgcn_drop_out':[0.5, 0.4, 0.5, 0.1, 0.5],
#                   'ffn_drop_out':[0., 0., 0.2, 0., 0.3],
#                   'lr':[0.0001, 0.003, 0.003, 0.001, 0.001],
#                   'classification':False}
# for i in range(5):
#      maskgnn_model_explain_for_no_mask(seed=i, task_name=task_name, rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'][i],
#                                      ffn_hidden_feats=hyperparameter['ffn_hidden_feats'][i],
#                                      lr=hyperparameter['lr'][i], classification=hyperparameter['classification'])


task_name = 'hERG'
hyperparameter = {'rgcn_hidden_feats':[[128, 128, 128], [64, 64, 64], [128, 128], [256, 256, 256], [64, 64]], 'ffn_hidden_feats':[128, 128, 128, 128, 64],
                'rgcn_drop_out':[0, 0.1, 0, 0.5, 0], 'ffn_drop_out':[0.1, 0.4, 0.2, 0.4, 0.2], 'lr':[0.003, 0.001, 0.003, 0.0003, 0.0003],
                'classification':True}
for i in range(5):
     maskgnn_model_explain_for_no_mask(seed=i, task_name=task_name, rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'][i],
                                     ffn_hidden_feats=hyperparameter['ffn_hidden_feats'][i],
                                     lr=hyperparameter['lr'][i], classification=hyperparameter['classification'])












