import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import matplotlib.colors as mcolors
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Geometry
rdDepictor.SetPreferCoordGen(True)


def return_rings_attribution(mol, sub_index, sub_atom_attribution_list, fg=False):
    ring_index = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
    # print(sub_index, sub_atom_attribution_list)
    ring_attribution_index = []

    # 官能团时，没有返回所有原子的attribution,因此需要重新对应
    if fg:
        ring_attribution = []
        sub_atom_count = 0
        for sub in sub_index:
            ring_attribution.append(sub_atom_attribution_list[sub_atom_count])
            sub_atom_count = sub_atom_count + len(sub)

    # 判断环在哪个子结构中，并返回对应的attribution
    final_ring_index = []
    for ring in ring_index:
        for i, sub in enumerate(sub_index):
            if len(list(set(ring).intersection(set(sub)))) > 0:
                if fg:
                    ring_attribution_index.append(ring_attribution[i])
                    final_ring_index.append(ring)
                else:
                    ring_attribution_index.append(sub_atom_attribution_list[sub[0]])
                    final_ring_index.append(ring)
    return final_ring_index, ring_attribution_index


sns.set(color_codes=True)

def atom_attribution_visualize(smiles, atom_attribution, cmap_name='seismic_r',):
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    highlight_atom_colors = {}
    atomrads = {}

    for i in range(mol.GetNumAtoms()):
        highlight_atom_colors[i] = [plt_colors.to_rgba(float(atom_attribution[i]))]
        atomrads[i] = 0.2
    rdDepictor.Compute2DCoords(mol)

    # now draw the molecule, with highlights:
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()
    drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, {}, atomrads, {})
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    drawer.WriteDrawingText('{}_atom.png'.format(smiles))
    return png


def sub_attribution_visualize(smiles, atom_attribution, bond_attribution, ring_attribution, atom_list=None, bond_list=None,
                              ring_list=None, cmap_name='seismic_r', sub_type='brics'):
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1) 
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    highlight_atom_colors = {}
    highlight_bond_colors = {}
    atomrads = {}
    widthmults = {}
    if atom_list is None:
        atom_list = range(0, mol.GetNumAtoms())
    if bond_list is None:
        bond_list = range(0, mol.GetNumBonds())
    for i in atom_list:
        highlight_atom_colors[i] = [plt_colors.to_rgba(float(atom_attribution[atom_list.index(i)]))]
        atomrads[atom_list.index(i)] = 0.2
    if len(bond_list) > 0:
        for i in bond_list:
            highlight_bond_colors[i] = [plt_colors.to_rgba(float(bond_attribution[bond_list.index(i)]))]
            widthmults[bond_list.index(i)] = 1
    if len(ring_list)>0:
        ring_color = [plt_colors.to_rgba(float(ring_attribution[i])) for i in range(len(ring_list))]
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()
#     drawer.SetFontSize(1)

    if len(ring_list)>0:
        # a hack to set the molecule scale
        drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, highlight_bond_colors, atomrads, widthmults)
        drawer.ClearDrawing()
        conf = mol.GetConformer()
        for i in range(len(ring_list)):
            aring = ring_list[i]
            ring_colors_i = ring_color[i]
            ps = []
            for aidx in aring:
                pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
                ps.append(pos)
            drawer.SetFillPolys(True)
            drawer.SetColour(ring_colors_i)
            drawer.DrawPolygon(ps)
        dos.clearBackground = False

    # now draw the molecule, with highlights:
    drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, highlight_bond_colors, atomrads, widthmults)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    drawer.WriteDrawingText('{}_{}.png'.format(smiles, sub_type))
    return png


# 根据每个smiles的index，得到其mask的子结构的index
def return_smask_index_i(smask_index_list, smi_index):
    smask_index_i = [smask_index_list[i] for i in smi_index]
    return smask_index_i


def return_bond_sub_index(bond_idx, smask):
    bond_sub_index = []
    for bond_atom_idx in bond_idx:
        for i, sub in enumerate(smask):
            if bond_atom_idx in sub:
                bond_sub_index.append(i)
                break
    if len(bond_sub_index) < 2:
        bond_attribution_index = -1
    elif bond_sub_index[0] == bond_sub_index[1]:
        bond_attribution_index = bond_sub_index[0]
    else:
        bond_attribution_index = -1
    return bond_attribution_index


def return_atom_and_sub_attribution(smiles, smask_index, attribution, fg_smask_index, fg_attribution,
                                    murcko_smask_index, murcko_attribution):
    mol = Chem.MolFromSmiles(smiles)
    n_atom = mol.GetNumAtoms()
    atom_attribution_list = attribution[:n_atom]
    remain_smask = smask_index[n_atom:]
    remain_attribution = attribution[n_atom:]
    sub_attribution_list = [0 for x in range(n_atom)]
    # 为brics子结构每个原子添加attribution
    for i, sub in enumerate(remain_smask):
        for atom_index in sub:
            sub_attribution_list[atom_index] = remain_attribution[i]

    # 为brics子结构每个环添加attribution
    brics_ring_list, brics_ring_attribution_list = return_rings_attribution(mol, remain_smask, sub_attribution_list)
    # 为子结构的键添加attribution
    # 为所有键添加子结构的index，为键找到对应子结构，不在子结构中的键设置为-1
    bond_attribution_index_list = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bond_idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        bond_attribution_index_list.append(return_bond_sub_index(bond_idx, remain_smask))

    # 为brics子结构找出包含的键，并给出attribution
    bond_attribution_list = []
    bond_list = []
    for i, bond_attribution_index in enumerate(bond_attribution_index_list):
        if bond_attribution_index == -1:
            pass
        else:
            bond_list.append(i)
            bond_attribution_list.append(remain_attribution[bond_attribution_index])

    # 为fg子结构每个原子添加attribution
    fg_attribution_list = []
    fg_atom_list = []
    for i, sub in enumerate(fg_smask_index):
        for atom_index in sub:
            fg_attribution_list.append(fg_attribution[i])
            fg_atom_list.append(atom_index)

    # 为fg子结构每个环添加attribution
    fg_ring_list, fg_ring_attribution_list = return_rings_attribution(mol, fg_smask_index, fg_attribution_list, fg=True)

    # 为fg子结构每个键添加attribution
    fg_bond_attribution_index_list = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bond_idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        fg_bond_attribution_index_list.append(return_bond_sub_index(bond_idx, fg_smask_index))

    # 为fg子结构找出包含的键，并给出attribution
    fg_bond_attribution_list = []
    fg_bond_list = []
    for i, bond_attribution_index in enumerate(fg_bond_attribution_index_list):
        if bond_attribution_index == -1:
            pass
        else:
            fg_bond_list.append(i)
            fg_bond_attribution_list.append(fg_attribution[bond_attribution_index])

    # 为murcko子结构每个原子添加attribution
    murcko_attribution_list = []
    murcko_atom_list = []
    for i, sub in enumerate(murcko_smask_index):
        for atom_index in sub:
            murcko_attribution_list.append(murcko_attribution[i])
            murcko_atom_list.append(atom_index)
    # 为murcko子结构每个环添加attribution
    murcko_ring_list, murcko_ring_attribution_list = return_rings_attribution(mol, murcko_smask_index, murcko_attribution_list)

    # 为murcko子结构每个键添加attribution
    murcko_bond_attribution_index_list = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bond_idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        murcko_bond_attribution_index_list.append(return_bond_sub_index(bond_idx, murcko_smask_index))

    # 为murcko子结构找出包含的键，并给出attribution
    murcko_bond_attribution_list = []
    murcko_bond_list = []
    for i, bond_attribution_index in enumerate(murcko_bond_attribution_index_list):
        if bond_attribution_index == -1:
            pass
        else:
            murcko_bond_list.append(i)
            murcko_bond_attribution_list.append(murcko_attribution[bond_attribution_index])
    return atom_attribution_list, sub_attribution_list, bond_attribution_list, bond_list, fg_attribution_list,\
           fg_atom_list, fg_bond_attribution_list, fg_bond_list, murcko_attribution_list, murcko_atom_list, \
           murcko_bond_attribution_list, murcko_bond_list, brics_ring_list, brics_ring_attribution_list,\
           murcko_ring_list, murcko_ring_attribution_list, fg_ring_list, fg_ring_attribution_list


def return_atom_and_sub_attribution_emerge(smiles, emerge_smask_index,
                                           emerge_attribution):
    mol = Chem.MolFromSmiles(smiles)
    # 为二分子结构每个原子添加attribution
    emerge_attribution_list = []
    emerge_atom_list = []
    emerge_smask_index = [emerge_smask_index]
    emerge_attribution = [emerge_attribution]
    for i, sub in enumerate(emerge_smask_index):
        for atom_index in sub:
            emerge_attribution_list.append(emerge_attribution[i])
            emerge_atom_list.append(atom_index)
    # 为emerge子结构每个环添加attribution
    emerge_ring_list, emerge_ring_attribution_list = return_rings_attribution(mol, emerge_smask_index, emerge_attribution_list)

    # 为emerge子结构每个键添加attribution
    emerge_bond_attribution_index_list = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bond_idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        emerge_bond_attribution_index_list.append(return_bond_sub_index(bond_idx, emerge_smask_index))

    # 为emerge子结构找出包含的键，并给出attribution
    emerge_bond_attribution_list = []
    emerge_bond_list = []
    for i, bond_attribution_index in enumerate(emerge_bond_attribution_index_list):
        if bond_attribution_index == -1:
            pass
        else:
            emerge_bond_list.append(i)
            emerge_bond_attribution_list.append(emerge_attribution[bond_attribution_index])
    return emerge_attribution_list, emerge_atom_list, emerge_bond_attribution_list, emerge_bond_list, emerge_ring_attribution_list, emerge_ring_list