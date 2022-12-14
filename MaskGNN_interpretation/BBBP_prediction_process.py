import numpy as np
from MaskGNN_interpretation import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from MaskGNN_interpretation.maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
import pickle as pkl
import time


task_name_list = ['BBBP_TPSA', 'BBBP_LogP', 'BBBP_MW', 'BBBP_HBDs', 'BBBP_MW']

# fix parameters of model
def SMEG_explain_for_substructure(seed, task_name, rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128,
                                  lr=0.0003, classification=True, sub_type='fg'):
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
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = task_name  # change
    args['bin_path'] = '../data/graph_data/{}_for_{}.bin'.format(args['data_name'], sub_type)
    args['group_path'] = '../data/graph_data/{}_group_for_{}.csv'.format(args['data_name'], sub_type)
    args['smask_path'] = '../data/graph_data/{}_smask_for_{}.npy'.format(args['data_name'], sub_type)
    args['seed'] = seed

    print('***************************************************************************************************')
    print('{}, seed {}, substructure type {}'.format(args['task_name'], args['seed'] + 1, sub_type))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        smask_path=args['smask_path'],
        classification=args['classification'],
        random_shuffle=False
    )
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
        pos_weight_np = pos_weight(train_set)
        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none',
                                                    pos_weight=pos_weight_np.to(args['device']))
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
    pred_name = '{}_{}_{}'.format(args['task_name'], sub_type, seed + 1)
    stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion,
                                          out_path='../prediction/{}/{}_test'.format(sub_type, pred_name))
    stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion,
                                           out_path='../prediction/{}/{}_train'.format(sub_type, pred_name))
    stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion,
                                         out_path='../prediction/{}/{}_val'.format(sub_type, pred_name))
    print('Mask prediction is generated!')



for task in task_name_list:
    for sub_type in ['fg']:
        # load
        with open('../result/hyperparameter_{}.pkl'.format(task), 'rb') as f:
            hyperparameter = pkl.load(f)
        for i in range(10):
            SMEG_explain_for_substructure(seed=i, task_name=task,
                                          rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                          ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                          lr=hyperparameter['lr'], classification=hyperparameter['classification'],
                                          sub_type=sub_type)



import os
task_list = task_name_list
sub_type_list = ['fg', 'mol']

for task_name in task_list:
    for sub_type in sub_type_list:
        try:
            # 将训练集，验证集，测试集数据合并
            result_summary = pd.DataFrame()
            for i in range(10):
                seed = i + 1
                result_train = pd.read_csv('../prediction/{}/{}_{}_{}_train_prediction.csv'.format(sub_type, task_name, sub_type, seed))
                result_val = pd.read_csv('../prediction/{}/{}_{}_{}_val_prediction.csv'.format(sub_type, task_name, sub_type, seed))
                result_test = pd.read_csv('../prediction/{}/{}_{}_{}_test_prediction.csv'.format(sub_type, task_name, sub_type, seed))
                group_list = ['training' for x in range(len(result_train))] + ['val' for x in range(len(result_val))] + ['test' for
                                                                                                                         x in range(
                        len(result_test))]
                result = pd.concat([result_train, result_val, result_test], axis=0)
                # mol是模型最初预测的时候给的结果，batch是会随机乱序的，所以需要重新排序
                result['group'] = group_list
                if sub_type == 'mol':
                    result.sort_values(by='smiles', inplace=True)
                # 合并五个随机种子结果，并统计方差和均值
                if seed == 1:
                    result_summary['smiles'] = result['smiles']
                    result_summary['label'] = result['label']
                    result_summary['sub_name'] = result['sub_name']
                    result_summary['group'] = result['group']
                    result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
                if seed > 1:
                    result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
                print('{} {} sum succeed.'.format(task_name, sub_type))
            pred_columnms = ['pred_{}'.format(i+1) for i in range(10)]
            data_pred = result_summary[pred_columnms]
            result_summary['pred_mean'] = data_pred.mean(axis=1)
            result_summary['pred_std'] = data_pred.std(axis=1)
            dirs = '../prediction/summary/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            result_summary.to_csv('../prediction/summary/{}_{}_prediction_summary.csv'.format(task_name, sub_type), index=False)
        except:
            print('{} {} sum failed.'.format(task_name, sub_type))


import os


for task_name in task_name_list:
    for sub_type in ['fg']:
        attribution_result = pd.DataFrame()
        print('{} {}'.format(task_name, sub_type))
        result_sub = pd.read_csv('../prediction/summary/{}_{}_prediction_summary.csv'.format(task_name, sub_type))
        result_mol = pd.read_csv('../prediction/summary/{}_{}_prediction_summary.csv'.format(task_name, 'mol'))
        mol_pred_mean_list_for_sub = [result_mol[result_mol['smiles'] == smi]['pred_mean'].tolist()[0] for smi in
                                 result_sub['smiles'].tolist()]
        mol_pred_std_list_for_sub = [result_mol[result_mol['smiles'] == smi]['pred_std'].tolist()[0] for smi in
                                 result_sub['smiles'].tolist()]
        attribution_result['smiles'] = result_sub['smiles']
        attribution_result['label'] = result_sub['label']
        attribution_result['sub_name'] = result_sub['sub_name']
        attribution_result['group'] = result_sub['group']
        attribution_result['sub_pred_mean'] = result_sub['pred_mean']
        attribution_result['sub_pred_std'] = result_sub['pred_std']
        attribution_result['mol_pred_mean'] = mol_pred_mean_list_for_sub
        attribution_result['mol_pred_std'] = mol_pred_std_list_for_sub
        sub_pred_std_list = result_sub['pred_std']
        attribution_result['attribution'] = attribution_result['mol_pred_mean'] - attribution_result['sub_pred_mean']
        attribution_result['attribution_normalized'] = (np.exp(attribution_result['attribution'].values) - np.exp(
            -attribution_result['attribution'].values)) / (np.exp(attribution_result['attribution'].values) + np.exp(
            -attribution_result['attribution'].values))
        dirs = '../prediction/attribution/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        attribution_result.to_csv('../prediction/attribution/{}_{}_attribution_summary.csv'.format(task_name, sub_type), index=False)


def prob2label(prob):
    if prob<0.5:
        return 0
    else:
        return 1


task_name_list = task_name_list
for task_name in task_name_list:
    result = pd.read_csv('../prediction/attribution/{}_fg_attribution_summary.csv'.format(task_name))
    # result = result[result['group']=='training']
    fg_list = list(set(result['sub_name'].tolist()))
    fg_list.sort()
    print(len(fg_list), fg_list)
    average_attribution_fg = pd.DataFrame()
    mol_num_list = []
    mean_att_list = []
    for i, fg in enumerate(fg_list):
        result_fg = result[result['sub_name']==fg]
        attribution_mean = result_fg['attribution'].mean()
        # 正负改过来，对应毒性问题时，正值是利于毒性，为了可视化，将其改为负值
        pred_labels = [prob2label(prob) for prob in result_fg['mol_pred_mean'].tolist()]
        if len(result_fg)>10:
            print('**************************************************************************************')
            print("{} function group. number of mol: {}; attribution: {}".format(fg, len(result_fg), round(attribution_mean, 4)))
            print('**************************************************************************************')
            print()
        mol_num_list.append(len(result_fg))
        mean_att_list.append(round(attribution_mean, 4))

    print()
    average_attribution_fg['sub_name'] = [fg for fg in fg_list]
    average_attribution_fg['mol_num'] = mol_num_list
    average_attribution_fg['attribution_mean'] = mean_att_list
    average_attribution_fg.sort_values(by=['attribution_mean'], inplace=True)
    average_attribution_fg.to_csv('../prediction/A_{}_average_attribution_summary.csv'.format(task_name), index=False)










