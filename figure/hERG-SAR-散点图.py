#调用seaborn
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

sns.set(font_scale=1.25, style='white')

#调用seaborn自带数据集
task_name = 'hERG SAR'
data_esol = pd.read_csv('../prediction/A_{}_average_attribution_summary.csv'.format('ESOL'))
data_hERG = pd.read_csv('../prediction/A_{}_average_attribution_summary.csv'.format('hERG'))
esol_sub_name_list = data_esol.sub_name.tolist()
herg_sar_data = pd.DataFrame()
data_hERG_attribution = []
data_hERG_mol_num = []
for sub_name in esol_sub_name_list:
    data_hERG_attribution.append(data_hERG[data_hERG['sub_name']==sub_name].attribution_mean.item())
    data_hERG_mol_num.append(data_hERG[data_hERG['sub_name']==sub_name].mol_num.item())

herg_sar_data['sub_name'] = data_esol.sub_name.tolist()
herg_sar_data['esol attribution'] = data_esol.attribution_mean.tolist()
herg_sar_data['esol_mol_num'] = data_esol.mol_num.tolist()
herg_sar_data['hERG attribution'] = data_hERG_attribution
herg_sar_data['hERG_mol_num'] = data_hERG_mol_num
herg_sar_data = herg_sar_data[herg_sar_data['hERG_mol_num']>10]
herg_sar_data = herg_sar_data[herg_sar_data['esol_mol_num']>10]
herg_sar_data.to_csv('hERG_SAR.csv', index=False)

#显示数据集
spearman,p = stats.spearmanr(herg_sar_data['esol attribution'], herg_sar_data['hERG attribution'])
# data['attribution'], data['pred_mean']
spearman = round(spearman,3)
p = '{:.3e}'.format(p)

print(spearman, p)

g = sns.jointplot(x='esol attribution',y='hERG attribution', data=herg_sar_data, height=5, kind='reg')
g.fig.set_figwidth(20)
g.fig.set_figheight(8)
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.ax_joint.text(
    s=f' spearmanr = {spearman}; p = {p} ',x=.75,y=.95,transform=g.ax_joint.transAxes,
    bbox={'boxstyle':'round','pad':0.6,'facecolor':'white','edgecolor':'gray'})
font2 = {'weight' : 'normal',
        'size'   : 15,
        }
plt.xlabel("The average attribution of ESOL's functional group", font2)
plt.ylabel("The average attribution of hERG's functional group", font2)
plt.savefig('{}_散点图.png'.format(task_name), dpi=300, bbox_inches='tight')
plt.show()