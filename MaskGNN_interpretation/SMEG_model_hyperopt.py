import numpy as np
import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle as pkl
import time


# fix parameters of model
def SMEG_hyperopt(times, task_name, max_evals=30, classification=False):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 256
    args['in_feats'] = 40
    args['max_evals'] = max_evals
    args['classification'] = classification
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = task_name  # change
    args['bin_path'] = '../data/graph_data/' + args['data_name'] + '.bin'
    args['group_path'] = '../data/graph_data/' + args['data_name'] + '_group.csv'
    args['times'] = times

    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    if args['classification']:
        result_pd['index'] = ['accuracy', 'sensitivity', 'specificity', 'f1-score', 'precision', 'recall', 'error rate', 'mcc']
        args['metric_name'] = 'accuracy'
        args['mode'] = 'higher'
    else:
        result_pd['index'] = ['r2', 'mae', 'rmse']
        args['metric_name'] = 'r2'
        args['mode'] = 'higher'

    space = {'rgcn_hidden_feats': hp.choice('rgcn_hidden_feats',
                                            [[64, 64], [128, 128], [256, 256], [64, 64, 64], [128, 128, 128],
                                             [256, 256, 256]]),
             'ffn_hidden_feats': hp.choice('ffn_hidden_feats', [64, 128, 256]),
             'ffn_drop_out': hp.choice('ffn_drop_out', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
             'rgcn_drop_out': hp.choice('rgcn_drop_out', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
             'lr': hp.choice('lr', [0.003, 0.001, 0.0003, 0.0001]),
             }

    train_set, val_set, test_set, task_number = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        classification=args['classification'],
        seed=2022,
        random_shuffle=False
    )

    print("Molecule graph is loaded!")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
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
    
    def hyperopt_my_rgcn(parameter):
        model = RGCN(ffn_hidden_feats=parameter['ffn_hidden_feats'],
                     ffn_dropout=parameter['ffn_drop_out'],
                     rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=parameter['rgcn_hidden_feats'],
                     rgcn_drop_out=parameter['rgcn_drop_out'],
                     classification=args['classification'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
        model.to(args['device'])
        for epoch in range(args['num_epochs']):
            # Train
            lr = parameter['lr']
            optimizer = Adam(model.parameters(), lr=lr)
            _, total_loss = run_a_train_epoch(args, model, train_loader, loss_criterion, optimizer)
            # Validation and early stop
            train_score, trian_loss = run_an_eval_epoch(args, model, train_loader, loss_criterion, out_path=None)
            val_score, val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion, out_path=None)
            early_stop = stopper.step(val_score[0], model)
            print('epoch {:d}/{:d}, {}, lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}  train: {:.4f}, valid: {:.4f}, best valid score {:.4f}'.format(
                  epoch + 1, args['num_epochs'], args['metric_name'], lr, total_loss, val_loss, train_score[0], val_score[0],
                  stopper.best_score))
            if early_stop:
                break
        stopper.load_checkpoint(model)
        val_score = -(run_an_eval_epoch(args, model, val_loader, loss_criterion, out_path=None)[0][0])
        return {'loss': val_score, 'status': STATUS_OK, 'model': model}

    # hyper parameter optimization
    trials = Trials()
    best = fmin(hyperopt_my_rgcn, space, algo=tpe.suggest, trials=trials, max_evals=args['max_evals'])
    print(best)

    # load the best model parameters
    args['rgcn_hidden_feats'] = [[64, 64], [128, 128], [256, 256], [64, 64, 64], [128, 128, 128], [256, 256, 256]][
        best['rgcn_hidden_feats']]
    args['ffn_hidden_feats'] = [64, 128, 256][best['ffn_hidden_feats']]
    args['rgcn_drop_out'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5][best['rgcn_drop_out']]
    args['ffn_drop_out'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5][best['ffn_drop_out']]
    args['lr'] = [0.003, 0.001, 0.0003, 0.0001][best['lr']]

    for time_id in range(args['times']):
        set_random_seed(2022+time_id*10)
        one_time_train_result = []
        one_time_val_result = []
        one_time_test_result = []
        print('***************************************************************************************************')
        print('{}, {}/{} time'.format(args['task_name'], time_id + 1,
                                         args['times']))
        print('***************************************************************************************************')
        model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                         ffn_dropout=args['ffn_drop_out'],
                         rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                         rgcn_drop_out=args['rgcn_drop_out'],
                         classification=args['classification'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name']+'_'+str(time_id+1), mode=args['mode'])
        model.to(args['device'])
        for epoch in range(args['num_epochs']):
            # Train
            lr = args['lr']
            optimizer = Adam(model.parameters(), lr=lr)
            _, total_loss = run_a_train_epoch(args, model, train_loader, loss_criterion, optimizer)
            # Validation and early stop
            train_score, trian_loss = run_an_eval_epoch(args, model, train_loader, loss_criterion, out_path=None)
            val_score, val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion, out_path=None)
            test_score, test_loss = run_an_eval_epoch(args, model, test_loader, loss_criterion, out_path=None)
            early_stop = stopper.step(val_score[0], model)
            print('epoch {:d}/{:d}, {}, lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}  train: {:.4f}, valid: {:.4f}, best valid score {:.4f}, '
                  'test: {:.4f}'.format(
                  epoch + 1, args['num_epochs'], args['metric_name'], lr, total_loss, val_loss, train_score[0], val_score[0],
                  stopper.best_score, test_score[0]))
            if early_stop:
                break
        stopper.load_checkpoint(model)
        train_score = run_an_eval_epoch(args, model, train_loader, loss_criterion, out_path=None)[0][0]
        val_score = run_an_eval_epoch(args, model, val_loader, loss_criterion, out_path=None)[0][0]
        test_score = run_an_eval_epoch(args, model, test_loader, loss_criterion, out_path=None)[0][0]
        pred_name = 'mol_{}'.format(time_id + 1)
        stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion,
                                               out_path='../prediction/mol/' + args['task_name'] + '_' + pred_name + '_test')
        stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion,
                                                out_path='../prediction/mol/' + args['task_name'] + '_' + pred_name + '_train')
        stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion,
                                              out_path='../prediction/mol/' + args['task_name'] + '_' + pred_name + '_val')
        result_pd['train_' + str(time_id + 1)] = stop_train_list
        result_pd['val_' + str(time_id + 1)] = stop_val_list
        result_pd['test_' + str(time_id + 1)] = stop_test_list
        print(result_pd[['index', 'train_' + str(time_id + 1), 'val_' + str(time_id + 1), 'test_' + str(time_id + 1)]])
        print('********************************{}, {}_times_result*******************************'.format(args['task_name'],
                                                                                                          time_id + 1))
        print("training_result:", round(train_score, 4))
        print("val_result:", round(val_score, 4))
        print("test_result:", round(test_score, 4))

        one_time_train_result.append(train_score)
        one_time_val_result.append(val_score)
        one_time_test_result.append(test_score)
        # except:
        #     task_number = task_number - 1
        all_times_train_result.append(round(np.array(one_time_train_result).mean(), 4))
        all_times_val_result.append(round(np.array(one_time_val_result).mean(), 4))
        all_times_test_result.append(round(np.array(one_time_test_result).mean(), 4))
        # except:
        #     print('{} times is failed!'.format(time_id+1))
        print("************************************{}_times_result************************************".format(
            time_id + 1))
        print('the train result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_train_result))
        print('the average train result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                          np.array(all_times_train_result).mean()))
        print('the train result of all tasks (std): {:.3f}'.format(np.array(all_times_train_result).std()))
        print('the train result of all tasks (var): {:.3f}'.format(np.array(all_times_train_result).var()))

        print('the val result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_val_result))
        print('the average val result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                        np.array(all_times_val_result).mean()))
        print('the val result of all tasks (std): {:.3f}'.format(np.array(all_times_val_result).std()))
        print('the val result of all tasks (var): {:.3f}'.format(np.array(all_times_val_result).var()))

        print('the test result of all tasks ({}):'.format(args['metric_name']), np.array(all_times_test_result))
        print('the average test result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                         np.array(all_times_test_result).mean()))
        print('the test result of all tasks (std): {:.3f}'.format(np.array(all_times_test_result).std()))
        print('the test result of all tasks (var): {:.3f}'.format(np.array(all_times_test_result).var()))
    with open('../result/hyperparameter_{}.pkl'.format(task_name), 'wb') as f:
        pkl.dump(args, f, pkl.HIGHEST_PROTOCOL)
    result_pd.to_csv('../result/SMEG_' + args['task_name'] + '_all_result.csv', index=False)



















