from SMEG_model_hyperopt import SMEG_hyperopt
from maskgnn import set_random_seed
import argparse
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_random_seed(10)
    regression_task_list = ['ESOL']
    classification_task_list = ['Mutagenicity', 'hERG']
    for task in regression_task_list:
        SMEG_hyperopt(10, task, 30, classification=False)
    for task in classification_task_list:
        SMEG_hyperopt(10, task, 30, classification=True)