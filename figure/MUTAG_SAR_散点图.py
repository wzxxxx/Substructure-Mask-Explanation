#调用seaborn
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

sns.set(font_scale=2, style='ticks')

#调用seaborn自带数据集
task_name = 'Mutagenicity'
data = pd.read_csv('A_{}_FG_SAR_OPT.csv'.format(task_name))
#显示数据集

pearson,p = stats.pearsonr(data['pred_mean'], data['attribution'])
# data['attribution'], data['pred_mean']
pearson = round(pearson,3)
p = '{:.3e}'.format(p)
print(pearson, p)

g = sns.jointplot(x='attribution',y='pred_mean', data=data, height=5, kind='reg')
g.ax_marg_x.remove()
g.ax_marg_y.remove()
g.ax_joint.text(
    s=f' pearsonr = {pearson}; p = {p} ',x=.05,y=.95,transform=g.ax_joint.transAxes,
    bbox={'boxstyle':'round','pad':0.25,'facecolor':'white','edgecolor':'gray'})
font2 = {'weight' : 'normal',
        'size'   : 25,
        }
plt.xlabel('Function group attribution', font2)
plt.ylabel('The prediction of the model', font2)
plt.savefig('{}_散点图.png'.format(task_name), dpi=300, bbox_inches='tight')
plt.show()