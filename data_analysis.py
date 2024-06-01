# %%
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# %%

df = pd.read_csv('bioactivity_preprocessed_data.csv')

def lipinski(smiles, verbose = False):
    moldata = []
    for elem in smiles:
        mol =  Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1,1)
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
# %%
df_lipinski = lipinski(df.canonical_smiles)
df_combined = pd.concat([df,df_lipinski], axis = 1)

def AC50(input):
    pAC50 = []

    for i in input['standard_value_norm']:
        molar = i*10**-9
        pAC50.append(-np.log10(molar))

        input['pAC50'] = pAC50
        x = input.drop('standard_value_norm', 1)
    
        return x  
# %%
df_combined.standard_value.describe()
# %%
def norm_value(input):
    norm = []

    for i in input['standard value']:
        if i > 100000000:
            i = 100000000
        norm.append(i)
    
    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)

    return x
# %%
df_norm = norm_value(df_combined)
df_final = pAC50(df_norm)
df_final.pAC50.describe()
