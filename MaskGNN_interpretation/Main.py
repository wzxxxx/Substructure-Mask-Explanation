from SMEG_model_hyperopt import SMEG_hyperopt
from maskgnn import set_random_seed
import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Develop RGCN models')
parser.add_argument('--task_name', type=str, help='the task name')
args = parser.parse_args()

if __name__ == '__main__':
    task = args.task_name
    set_random_seed(10)
    regression_task_list = ['ESOL', 'lipop']
    classification_task_list = ['Mutagenicity', 'hERG', 'BBBP']
    if task in regression_task_list:
        SMEG_hyperopt(10, task, 30, classification=False)
    if task in classification_task_list:
        SMEG_hyperopt(10, task, 30, classification=True)