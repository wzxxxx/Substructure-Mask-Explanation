import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
colors = sns.color_palette('tab10', 2)
for task_name in ['ESOL_top20', 'ESOL']:
        data = pd.read_csv('../prediction/summary/{}_6_brics_mol_mol_prediction_summary.csv'.format(task_name))
        ax = sns.kdeplot(data=data, x="pred_mean",hue="label", palette=colors, shade=True, legend=True)
        # plt.legend(loc='upper right')
        plt.legend(labels=['Hydrophilic BRICS fragments', 'Hydrophobic BRICS fragments'], loc='upper left')
        plt.subplots_adjust(bottom=0.1)

        font2 = {'weight' : 'normal',
                'size'   : 15,
                }
        plt.xlabel('Prediction of the ESOL models', font2)
        plt.ylabel('', font2)
        plt.savefig('{}_brics_分布图.png'.format(task_name), dpi=300, bbox_inches='tight')
        plt.close()
