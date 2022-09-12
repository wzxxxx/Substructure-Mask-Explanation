from scipy.stats import spearmanr
import pandas as pd

data = pd.read_csv('A_ESOL_FG_SAR_OPT.csv')
spearman_result = spearmanr(data['attribution'], data['pred_mean'])
print(spearman_result)
