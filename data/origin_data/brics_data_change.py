import pandas as pd

for task in ['ESOL', 'hERG', 'Mutagenicity']:
    data = pd.read_csv('../../brics_build_mol/{}_6_brics_mol.csv'.format(task))
    group = ['test' for x in range(len(data))]
    data['group'] = group
    data['{}_6_brics_mol'.format(task)]=data['label']
    data.to_csv('{}_6_brics_mol.csv'.format(task), index=False)