import pickle as pkl
with open('hyperparameter_{}.pkl'.format('ESOL'), 'rb') as f:
    hyperparameter = pkl.load(f)
    print(hyperparameter)