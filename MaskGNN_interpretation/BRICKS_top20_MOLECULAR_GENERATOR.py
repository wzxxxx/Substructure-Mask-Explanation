import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import re
import random
from itertools import combinations

def return_brics_leaf_structure(smiles):
    m = Chem.MolFromSmiles(smiles)
    res = list(BRICS.FindBRICSBonds(m))  # [((1, 2), ('1', '3000'))]
    # return brics_bond
    all_brics_bond = [set(res[i][0]) for i in range(len(res))]
    all_brics_substructure_subset = dict()
    # return atom in all_brics_bond
    all_brics_atom = []
    for brics_bond in all_brics_bond:
        all_brics_atom = list(set(all_brics_atom + list(brics_bond)))

    if len(all_brics_atom) > 0:
        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for brics_atom in all_brics_atom:
            brics_break_atom = []
            for brics_bond in all_brics_bond:
                if brics_atom in brics_bond:
                    brics_break_atom += list(set(brics_bond))
            brics_break_atom = [x for x in brics_break_atom if x != brics_atom]
            all_break_atom[brics_atom] = brics_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
    else:
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(m.GetNumAtoms())]
    all_brics_substructure_subset['substructure'] = substrate_idx
    all_brics_substructure_subset['substructure_bond'] = all_brics_bond
    return all_brics_substructure_subset


def return_match_brics_fragment(smiles, data):
    """
    输入： smiles: smiles
          data: attribution data
    brics_leaf_structure为return_brics_leaf_structure(smiles)生成的字典，
    此函数是为了将BRICS.BreakBRICSBonds()打碎的BRICS碎片与我们预测出来的结果对应起来。
    为重新生成的brics碎片附加attribution，需要将两者重新对应起来。
    由于brics_leaf_structure中的碎片key排序，就是碎片应有的排序，
    所以只需按照brics_leaf_structure重新对brics碎片进行排序就行。
    返回两个列表，碎片的smiles形式，以及碎片的attribution
    """
    brics_leaf_structure = return_brics_leaf_structure(smiles)
    mol = Chem.MolFromSmiles(smiles)
    atom_num = mol.GetNumAtoms()
    brics_leaf_structure_sorted_id = sorted(range(len(brics_leaf_structure['substructure'].keys())),
                                            key=lambda k: list(brics_leaf_structure['substructure'].keys())[k],
                                            reverse=False)
    frags_attribution = data[data['smiles'] == smiles].attribution.tolist()[atom_num:]
    m2 = BRICS.BreakBRICSBonds(mol)
    frags = Chem.GetMolFrags(m2, asMols=True)
    frags_smi = [Chem.MolToSmiles(x, True) for x in frags]
    sorted_frags_smi = [i for _, i in sorted(zip(list(brics_leaf_structure_sorted_id), frags_smi), reverse=False)]
    if len(sorted_frags_smi) != len(frags_attribution):
        sorted_frags_smi = []
        frags_attribution = []
    return sorted_frags_smi, frags_attribution


def return_rogue_smi(smiles_frag):
    # 将frag_smiles 删去连接部分，补充成完整的分子
    rogue_frag = re.sub('\[[0-9]+\*\]', '', smiles_frag)  # remove link atom
    return rogue_frag


def brics_mol_generator(frag_num=1, same_frag_combination_mol_num=1, mol_number=1, seed=2022, frags_list=None):
    fragms = [Chem.MolFromSmiles(x) for x in sorted(frags_list)]
    index_list = [x for x in range(len(fragms))]
    random.seed(seed)
    tried_valid_combnination_list = []
    all_tried_combnination_list = []
    all_generator_mol_smi = []
    for i in range(1000000):
        print('{} frag_num, {} mol is generator! {}/{} valid/all combination is tried.'.format(frag_num, len(all_generator_mol_smi), len(tried_valid_combnination_list), len(all_tried_combnination_list)))
        if len(all_generator_mol_smi)>mol_number:
            break
        random.shuffle(index_list)
        i_index_list = list(set(index_list[:frag_num]))
        if i_index_list not in all_tried_combnination_list:
            try:
                frag_combination = [fragms[index_list[x]] for x in range(frag_num)]
                ms = BRICS.BRICSBuild(frag_combination) # 每次随机选几个碎片来组成化合物
                generator_mol_i_list = [next(ms) for x in range(same_frag_combination_mol_num)] # 每个碎片组合形式生成几个分子
                [generator_mol_i.UpdatePropertyCache(strict=False) for generator_mol_i in generator_mol_i_list]# 对生成分子进行核对，重新计算化合价,环相关等属性
                valid_generator_smi_i_list = [Chem.MolToSmiles(mol) for mol in generator_mol_i_list]
                all_generator_mol_smi = all_generator_mol_smi + valid_generator_smi_i_list
                all_generator_mol_smi = list(set(all_generator_mol_smi))# 去除重复分子
                tried_valid_combnination_list.append(i_index_list)
                all_tried_combnination_list.append(i_index_list)
            except:
                all_tried_combnination_list.append(i_index_list)
                pass
    all_generator_mol_smi = all_generator_mol_smi[:mol_number]
    return all_generator_mol_smi


task_name_list = ['ESOL', 'Mutagenicity', 'hERG']
for task_name in task_name_list:
    brics_frags_data = pd.read_csv('../brics_build_mol/{}_brics_frag.csv'.format(task_name))
    # 让取到的分子是一次从attribution最大及最小来排序的
    brics_frags_data.sort_values(by='attribution', ascending=True, inplace=True)
    negative_brics_frags = brics_frags_data[brics_frags_data['attribution']<0].frag_smiles.tolist()
    brics_frags_data.sort_values(by='attribution', ascending=False, inplace=True)
    positive_brics_frags = brics_frags_data[brics_frags_data['attribution']>0].frag_smiles.tolist()
    print(0.2*len(negative_brics_frags))
    for frag_num in [6]:
        negative_brics_smi_list = brics_mol_generator(frag_num=frag_num, same_frag_combination_mol_num=10, mol_number=3000, seed=2022, frags_list=negative_brics_frags[:int(0.2*len(negative_brics_frags))])
        positive_brics_smi_list = brics_mol_generator(frag_num=frag_num, same_frag_combination_mol_num=10, mol_number=3000, seed=2022, frags_list=positive_brics_frags[:int(0.2*len(positive_brics_frags))])
        brics_smi = negative_brics_smi_list + positive_brics_smi_list
        labels = [-1 for x in range(3000)] + [1 for x in range(3000)]
        brics_mol_data = pd.DataFrame()
        brics_mol_data['smiles'] = brics_smi
        brics_mol_data['{}_top20_{}_brics_mol'.format(task_name, frag_num)] = labels
        brics_mol_data['group'] = ['test' for x in range(6000)]
        brics_mol_data.to_csv('../brics_build_mol/{}_top20_{}_brics_mol.csv'.format(task_name, frag_num), index=False)# 将生成的分子按照进行保存



