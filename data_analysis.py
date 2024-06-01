# %%
import pandas as pd
df = pd.read_csv('bioactivity_preprocessed_data.csv')
# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
# %%
def lipinski(smiles, verbose = False):
    moldata = []
    for elem in smiles:
        mol =  Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arrange(1,1)
    i = 0
    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                       desc_MolLogP,
                       desc_NumHDonors,
                       desc_NumAcceptors])
        if(i==0):
            baseData = row
        else:
            baseData = np.vstack([baseData,row])
        i = i+1

    columnNames = ["MW","LogP","NumHDonors","NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns = columnNames)

    return descriptors



    


    