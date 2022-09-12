import pandas as pd
from rdkit import Chem

task_name_list = ['ESOL', 'Mutagenicity', 'hERG']


def dataset_wash(task_name):
    # 清除无机分子， 混合物， 重复的分子
    # parameter
    # load data set
    data = pd.read_csv('{}.csv'.format(task_name))
    origin_data_num = len(data)

    # remove molecule can't processed by rdkit_des
    print('********dealing with compounds with rdkit_des*******')
    smiles_list = data['smiles'].values.tolist()
    cant_processed_smiles_list = []
    for index, smiles in enumerate(smiles_list):
        if index % 10000 == 0:
            print(index)
        try:
            molecule = Chem.MolFromSmiles(smiles)
            smiles_standard = Chem.MolToSmiles(molecule)
            data['smiles'][index] = smiles_standard
        except:
            cant_processed_smiles_list.append(smiles)
            data.drop(index=index, inplace=True)
    print("compounds can't be processed by rdkit_des: {} molecules, {}\n".format(len(cant_processed_smiles_list),
                                                                      cant_processed_smiles_list))

    # remove mixture and salt
    print('********dealing with inorganic compounds*******')
    data = data.reset_index(drop=True)
    smiles_list = data['smiles'].values.tolist()
    mixture_salt_list = []
    for index, smiles in enumerate(smiles_list):
        if index % 10000==0:
            print(index)
        symbol_list = list(smiles)
        if '.' in symbol_list:
            mixture_salt_list.append(smiles)
            data.drop(index=index, inplace=True)
    print('inorganic compounds: {} molecules, {}\n'.format(len(mixture_salt_list), mixture_salt_list))


    # remove inorganic compounds
    print('********dealing with inorganic compounds*******')
    data = data.reset_index(drop=True)
    smiles_list = data['smiles'].values.tolist()
    inorganics = []
    atom_list = []
    for index, smiles in enumerate(smiles_list):
        if index % 10000==0:
            print(index)
        mol = Chem.MolFromSmiles(smiles)
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                break
            else:
                count += 1
        if count == mol.GetNumAtoms():
            inorganics.append(smiles)
            data.drop(index=index, inplace=True)
    print('inorganic compounds: {} molecules, {}\n'.format(len(inorganics), inorganics))

    print('********dealing with duplicates*******')
    consistent_mol = ['smiles', task_name]
    print('duplicates:{} molecules {}\n'.format(len(data[data.duplicated(consistent_mol)]['smiles'].values),
                                                 data[data.duplicated(consistent_mol)]['smiles'].values))
    data.drop_duplicates(consistent_mol, keep='first', inplace=True)
    consistent_mol_2 = ['smiles']
    print('duplicates and conflict:{} molecules {}\n'.format(len(data[data.duplicated(consistent_mol_2)]['smiles'].values),
                                                                 data[data.duplicated(consistent_mol_2)]['smiles'].values))
    data.drop_duplicates(consistent_mol_2, keep=False, inplace=True)
    print('Data washing is over!')

    import random
    data = data[['smiles', task_name]]
    len_data = len(data)
    group_list = ['training' for x in range(int(len_data*0.8))] + ['valid' for j in range(int(len_data*0.1))] + ['test' for i in range(len_data - int(len_data*0.8)-int(len_data*0.1))]
    random.shuffle(group_list)
    data['group'] = group_list
    print("{} to {} after datawash.".format(origin_data_num, len_data))
    data.to_csv('{}.csv'.format(task_name), index=False)


for task in task_name_list:
    dataset_wash(task)