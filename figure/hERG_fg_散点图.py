import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def pro2label(prob):
    if prob < 0.5:
        return 0
    else:
        return 1

def attr2tag(attr):
    if attr < 0:
        return -1
    else:
        return 1


task_name = 'hERG'
plt.figure(figsize=(12, 30))
sns.set(font_scale=1.8, style='ticks')
colors = sns.color_palette('tab10', 2)
# sub_name,mol_num,attribution,attribution_var,eu_mean,eu_var,rmse,seed
data = pd.read_csv('../prediction/attribution/{}_fg_attribution_summary.csv'.format(task_name))
tag_list = [attr2tag(x) for x in data['attribution'].tolist()]
data['attribution_tag'] = tag_list
summary_data = pd.read_csv('../prediction/A_{}_average_attribution_summary.csv'.format(task_name))
sub_name_list = summary_data.sub_name.tolist()

print(len(summary_data))
# 删除FG数量小于10的分子
fg_rmse = []
# 同时计算出每一类别的准确度
for sub_name in sub_name_list:
    if len(data[data['sub_name']==sub_name])<=10:
        data = data[data['sub_name']!=sub_name]
        summary_data = summary_data[summary_data['sub_name']!=sub_name]
    else:
        sub_data = data[data['sub_name']==sub_name]
        label_s = sub_data['label'].tolist()
        pred_label = [pro2label(x) for x in sub_data['mol_pred_mean'].tolist()]
        fg_rmse.append(round((mean_squared_error(label_s, pred_label))**0.5, 3))

filter_sub_name_list = summary_data.sub_name.tolist()
summary_data.reset_index(inplace=True)
print(len(summary_data))

g = sns.stripplot(y="sub_name", x="attribution", hue='attribution_tag', palette=colors, data=data,  jitter=0.2, order=filter_sub_name_list, size=4)

atribution_x_cor=-1.05
for index, row in summary_data.iterrows():
    print(index)
    if row.attribution_mean<=0:
        g.text(atribution_x_cor, index+0.16, round(row.attribution_mean, 3), color="green",  ha="center")
        # g.text(index, -1.3, fg_rmse[index], color="blue",  ha="center") # 加准确率
    else:
        g.text(atribution_x_cor, index+0.16, round(row.attribution_mean, 3), color="red",  ha="center")
        # g.text(index, -1.3,  fg_rmse[index], color="blue",  ha="center") # 加准确率
g.text(atribution_x_cor, -0.8,  'Average \n Attribution', color="Black",  ha="center")


plt.subplots_adjust(bottom=0.4)
plt.xlim((-1.2, 1))
plt.xticks()
plt.legend([],[], frameon=False)

# sns.despine()
font2 = {'weight' : 'normal',
        'size'   : 25,
        }
plt.ylabel('Functional group', font2)
plt.xlabel('Attribution', font2)
plt.axvline(0, color="red") #竖线
plt.savefig('{}_fg_散点.png'.format(task_name), dpi=300, bbox_inches='tight')
plt.show()