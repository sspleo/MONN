from collections import Counter
from collections import defaultdict
import os
import pickle
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist


elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
             'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(),
                                        [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(),
                                        [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def Mol2Graph(mol):
    # convert molecule to GNN input
    def idxfunc(x): return x.GetIdx()

    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0

    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms,), dtype=np.int32)  # atom feature ID
    fbonds = np.zeros((n_bonds,), dtype=np.int32)  # bond feature ID
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    num_nbs_mat = np.zeros((n_atoms, max_nb), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_dict[''.join(
            str(x) for x in atom_features(atom).astype(int).tolist())]

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        fbonds[idx] = bond_dict[''.join(
            str(x) for x in bond_features(bond).astype(int).tolist())]
        try:
            atom_nb[a1, num_nbs[a1]] = a2
            atom_nb[a2, num_nbs[a2]] = a1
        except:
            return [], [], [], [], []
        bond_nb[a1, num_nbs[a1]] = idx
        bond_nb[a2, num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1

    for i in range(len(num_nbs)):
        num_nbs_mat[i, :num_nbs[i]] = 1

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat


def Batch_Mol2Graph(mol_list):
    res = list(map(lambda x: Mol2Graph(x), mol_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return fatom_list, fbond_list, gatom_list, gbond_list, nb_list


def Protein2Sequence(sequence, ngram=1):
    # convert sequence to CNN input
    sequence = sequence.upper()
    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(word_dict['X'])
        else:
            output.append(word_dict[word])
    if ngram == 3:
        output = [-1]+output+[-1]  # pad
    return np.array(output, np.int32)


def Batch_Protein2Sequence(sequence_list, ngram=3):
    res = list(map(lambda x: Protein2Sequence(x, ngram), sequence_list))
    return res


def get_mol_dict():
    if os.path.exists('../data/mol_dict'):
        with open('../data/mol_dict') as f:
            mol_dict = pickle.load(f)
    else:
        mol_dict = {}
        mols = Chem.SDMolSupplier('../data/Components-pub.sdf')
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        with open('../data/mol_dict', 'wb') as f:
            pickle.dump(mol_dict, f)
    # print('mol_dict',len(mol_dict))
    return mol_dict


def get_pairwise_label(pdbid, interaction_dict):
    if pdbid in interaction_dict:
        sdf_element = np.array([atom.GetSymbol().upper()
                                for atom in mol.GetAtoms()])
        atom_element = np.array(
            interaction_dict[pdbid]['atom_element'], dtype=str)
        atom_name_list = np.array(
            interaction_dict[pdbid]['atom_name'], dtype=str)
        atom_interact = np.array(
            interaction_dict[pdbid]['atom_interact'], dtype=int)
        nonH_position = np.where(atom_element != ('H'))[0]
        assert sum(atom_element[nonH_position] != sdf_element) == 0

        atom_name_list = atom_name_list[nonH_position].tolist()
        pairwise_mat = np.zeros((len(nonH_position), len(
            interaction_dict[pdbid]['uniprot_seq'])), dtype=np.int32)
        for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:
            atom_idx = atom_name_list.index(str(atom_name))
            assert atom_idx < len(nonH_position)

            seq_idx_list = []
            for seq_idx, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:
                if bond_type == bond_type_seq:
                    seq_idx_list.append(seq_idx)
                    pairwise_mat[atom_idx, seq_idx] = 1
        if len(np.where(pairwise_mat != 0)[0]) != 0:
            pairwise_mask = True
            return True, pairwise_mat
    return False, np.zeros((1, 1))


def get_fps(mol_list):
    fps = []
    for mol in mol_list:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=1024, useChirality=True)
        fps.append(fp)
    #print('fingerprint list',len(fps))
    return fps


def calculate_sims(fps1, fps2, simtype='tanimoto'):
    sim_mat = np.zeros((len(fps1), len(fps2)))  # ,dtype=np.float32)
    for i in range(len(fps1)):
        fp_i = fps1[i]
        if simtype == 'tanimoto':
            sims = DataStructs.BulkTanimotoSimilarity(fp_i, fps2)
        elif simtype == 'dice':
            sims = DataStructs.BulkDiceSimilarity(fp_i, fps2)
        sim_mat[i, :] = sims
    return sim_mat


def compound_clustering(ligand_list, mol_list):
    print 'start compound clustering...'
    fps = get_fps(mol_list)
    sim_mat = calculate_sims(fps, fps)
    #np.save('../preprocessing/'+MEASURE+'_compound_sim_mat.npy', sim_mat)
    print 'compound sim mat', sim_mat.shape
    C_dist = pdist(fps, 'jaccard')
    C_link = single(C_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        C_clusters = fcluster(C_link, thre, 'distance')
        len_list = []
        for i in range(1, max(C_clusters)+1):
            len_list.append(C_clusters.tolist().count(i))
        print 'thre', thre, 'total num of compounds', len(ligand_list), 'num of clusters', max(C_clusters), 'max length', max(len_list)
        C_cluster_dict = {ligand_list[i]: C_clusters[i]
                          for i in range(len(ligand_list))}
        with open('../preprocessing/'+MEASURE+'_compound_cluster_dict_'+str(thre), 'wb') as f:
            pickle.dump(C_cluster_dict, f, protocol=0)


def protein_clustering(protein_list, idx_list):
    print 'start protein clustering...'
    protein_sim_mat = np.load(
        '../data/pdbbind_protein_sim_mat.npy').astype(np.float32)
    sim_mat = protein_sim_mat[idx_list, :]
    sim_mat = sim_mat[:, idx_list]
    print 'original protein sim_mat', protein_sim_mat.shape, 'subset sim_mat', sim_mat.shape
    #np.save('../preprocessing/'+MEASURE+'_protein_sim_mat.npy', sim_mat)
    P_dist = []
    for i in range(sim_mat.shape[0]):
        P_dist += (1-sim_mat[i, (i+1):]).tolist()
    P_dist = np.array(P_dist)
    P_link = single(P_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        P_clusters = fcluster(P_link, thre, 'distance')
        len_list = []
        for i in range(1, max(P_clusters)+1):
            len_list.append(P_clusters.tolist().count(i))
        print 'thre', thre, 'total num of proteins', len(protein_list), 'num of clusters', max(P_clusters), 'max length', max(len_list)
        P_cluster_dict = {protein_list[i]: P_clusters[i]
                          for i in range(len(protein_list))}
        with open('../preprocessing/'+MEASURE+'_protein_cluster_dict_'+str(thre), 'wb') as f:
            pickle.dump(P_cluster_dict, f, protocol=0)


def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)


# Measure setting
MEASURE = 'KIKD'  # 'IC50' or 'KIKD'

# mol dict
mol_dict = get_mol_dict()
with open('../data/out7_final_pairwise_interaction_dict', 'rb') as f:
    interaction_dict = pickle.load(f)

wlnn_train_list = []
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
word_dict = defaultdict(lambda: len(word_dict))
for aa in aa_list:
    word_dict[aa]
word_dict['X']

i = 0
pair_info_dict = {}
f = open('../data/pdbbind_all_datafile.tsv')
print 'Step 2/5, generating labels...'
for line in f.readlines():
    i += 1
    if i % 1000 == 0:
        print 'processed sample num', i
    pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')
    # filter interaction type and invalid molecules
    if MEASURE == 'All':
        pass
    elif MEASURE == 'KIKD':
        if measure not in ['Ki', 'Kd']:
            continue
    elif measure != MEASURE:
        continue
    if cid not in mol_dict:
        print 'ligand not in mol_dict'
        continue
    mol = mol_dict[cid]

    # get labels
    value = float(label)
    pairwise_mask, pairwise_mat = get_pairwise_label(pdbid, interaction_dict)

    # handle the condition when multiple PDB entries have the same Uniprot ID and Inchi
    if inchi+' '+pid not in pair_info_dict:
        pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid,
                                         value, mol, seq, pairwise_mask, pairwise_mat]
    else:
        if pair_info_dict[inchi+' '+pid][6]:
            if pairwise_mask and pair_info_dict[inchi+' '+pid][3] < value:
                pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid,
                                                 value, mol, seq, pairwise_mask, pairwise_mat]
        else:
            if pair_info_dict[inchi+' '+pid][3] < value:
                pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid,
                                                 value, mol, seq, pairwise_mask, pairwise_mat]
f.close()


valid_value_list = []
valid_cid_list = []
valid_pid_list = []
valid_pairwise_mask_list = []
valid_pairwise_mat_list = []
mol_inputs, seq_inputs = [], []

# get inputs
for item in pair_info_dict:
    pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat = pair_info_dict[item]
    fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
    if fa == []:
        print 'num of neighbor > 6, ', cid
        continue
    mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
    seq_inputs.append(Protein2Sequence(seq, ngram=1))
    valid_value_list.append(value)
    valid_cid_list.append(cid)
    valid_pid_list.append(pid)
    valid_pairwise_mask_list.append(pairwise_mask)
    valid_pairwise_mat_list.append(pairwise_mat)
    wlnn_train_list.append(pdbid)


# [cid, pid] pair list
valid_pair_list = []
for i in range(len(valid_cid_list)):
    pair = [valid_cid_list[i], valid_pid_list[i]]
    valid_pair_list.append(pair)

# unique cid/pid list
unique_cid_list = list(set(valid_cid_list))
unique_pid_list = list(set(valid_pid_list))

# [cid,pid] interaction matrix
interaction_matrix = np.zeros((len(unique_cid_list), len(unique_pid_list)))
for i in range(len(valid_pair_list)):
    compound, protein = valid_pair_list[i]
    cid_ind = unique_cid_list.index(compound)
    pid_ind = unique_pid_list.index(protein)
    interaction_matrix[cid_ind, pid_ind] = 1

# compound interaction numbers
compound_int_num = np.sum(interaction_matrix, 1)
# protein interaction numbers
protein_int_num = np.sum(interaction_matrix, 0)

# make every np.float to np.int
compound_int_num = compound_int_num.astype(np.int64)
protein_int_num = protein_int_num.astype(np.int64)


def int_num_to_id(cid_or_pid, int_num):
    id_list = []
    if cid_or_pid == 'cid':
        arr = np.where(compound_int_num == int_num)[0]
        for i in range(len(arr)):
            cid = unique_cid_list[arr[i]]
            id_list.append(cid)
    elif cid_or_pid == 'pid':
        arr = np.where(protein_int_num == int_num)[0]
        for i in range(len(arr)):
            pid = unique_pid_list[arr[i]]
            id_list.append(pid)
    else:
        print 'error'
    return id_list


# get the max_compound_index
max_ind = np.where(compound_int_num == max(compound_int_num))
unique_cid_list[max_ind]


# numbers list

p = Counter(protein_int_num)
c = Counter(compound_int_num)

c_list = []
p_list = []
for i in range(len(c.keys())):
    c_list.append([c.keys()[i], c.values()[i]])

p_list = []
for i in range(len(p.keys())):
    p_list.append([p.keys()[i], p.values()[i]])


# make the element int and sort
for i in range(len(c_list)):
    c_list[i][0] = int(c_list[i][0])
c_list.sort()

for i in range(len(p_list)):
    p_list[i][0] = int(p_list[i][0])
p_list.sort()

# get first element of each sublist


def Extract(lst):
    return [item[0] for item in lst]

# get filtered interaction number list with threshold


def get_filtered_list(cid_or_pid, thre):
    if cid_or_pid == 'cid':
        filtered_list = [i for i in Extract(c_list) if i >= thre]
    elif cid_or_pid == 'pid':
        filtered_list = [i for i in Extract(p_list) if i >= thre]
    else:
        print 'error'
    return filtered_list

# get filtered id with threshold


def get_filtered_id(cid_or_pid, thre):
    flist = get_filtered_list(cid_or_pid, thre)
    id_dict = {}
    for i in range(len(flist)):
        id_dict[flist[i]] = int_num_to_id(cid_or_pid, flist[i])
    return id_dict
