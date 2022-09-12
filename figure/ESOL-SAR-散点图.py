#调用seaborn
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

sns.set(font_scale=1.25, style='white')

#调用seaborn自带数据集
task_name = 'ESOL SAR'
data_esol = pd.read_csv('../prediction/summary/{}_structure_optimization_mol_prediction_summary.csv'.format('ESOL'))
data_esol_attribution = pd.read_csv('../prediction/A_{}_average_attribution_summary.csv'.format('ESOL'))
print(len(data_esol_attribution))
esol_sub_name_list = data_esol.sub_name.tolist()
ESOL_sar_data = pd.DataFrame()
print(data_esol_attribution[data_esol_attribution['mol_num']>10])

ESOL_sar_data['sub_name'] = data_esol_attribution[data_esol_attribution['mol_num']>10]['sub_name'].tolist() + ['mol']
ESOL_sar_data['ESOL attribution'] = data_esol_attribution[data_esol_attribution['mol_num']>10]['attribution_mean'].tolist() + [-0.533]
print(ESOL_sar_data)
ESOL_sar_data['smiles'] = data_esol.smiles.tolist() + ['CC(C)C1CCC(C)CC1O']
ESOL_sar_data['prediction'] = data_esol.pred_mean.tolist() + [-2.53]


#显示数据集
spearman, p = stats.spearmanr(ESOL_sar_data['ESOL attribution'], ESOL_sar_data['prediction'])
# data['attribution'], data['pred_mean']
spearman = round(spearman,3)
p = '{:.3e}'.format(p)

print(spearman, p)

g = sns.jointplot(x='ESOL attribution',y='prediction', data=ESOL_sar_data, height=5, kind='reg')
g.fig.set_figwidth(20)
g.fig.set_figheight(8)
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.ax_joint.text(
    s=f' spearmanr = {spearman}; p = {p} ',x=.05,y=.95,transform=g.ax_joint.transAxes,
    bbox={'boxstyle':'round','pad':0.55,'facecolor':'white','edgecolor':'gray'})
font2 = {'weight' : 'normal',
        'size'   : 15,
        }
plt.xlabel("The average attribution of ESOL's functional group", font2)
plt.ylabel("The prediction of the ESOL model", font2)
plt.savefig('{}_sar_散点图.png'.format(task_name), dpi=300, bbox_inches='tight')
plt.show()