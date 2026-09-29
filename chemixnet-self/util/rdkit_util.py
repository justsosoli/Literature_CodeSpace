from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np

def get_maccs_key(smiles, labels):
    valid_maccs_list = []
    valid_label_list = []
    valid_smile_list = []
    for smile, label in zip(smiles, labels):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        maccs = MACCSkeys.GenMACCSKeys(mol)
        valid_smile_list.append(smile)
        valid_maccs_list.append(np.array(maccs))
        valid_label_list.append(label)
    valid_maccs_list = np.array(valid_maccs_list)
    valid_label_list = np.array(valid_label_list)
    valid_smile_list = np.array(valid_smile_list)
    return valid_smile_list, valid_maccs_list, valid_label_list
   
