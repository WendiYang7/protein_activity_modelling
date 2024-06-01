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

def pAC50(input_df):
    # Convert the normalized values to molar concentration
    input_df['molar'] = input_df['standard_value_norm'] * (10**-9)
    
    # Calculate the pAC50 values
    input_df['pAC50'] = -np.log10(input_df['molar'])
    
    # Drop the intermediate columns
    x = input_df.drop(columns=['standard_value_norm', 'molar'])
    
    return x
# %%
df_combined.standard_value.describe()
# %%
def norm_value(input_df):
    # Apply a function to each element in the 'standard_value' column to normalize it
    input_df['standard_value_norm'] = input_df['standard_value'].apply(lambda x: min(x, 100000000))
   
    # Drop the original 'standard_value' column
    x = input_df.drop(columns=['standard_value'])

    return x
# %%
df_norm = norm_value(df_combined)
df_final = pAC50(df_norm)
df_final.pAC50.describe()
# %%