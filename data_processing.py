# %%
# imports
import pandas as pd
from chembl_webresource_client.new_client import new_client

# %%
# target search for ERBB
target = new_client.target
target_query = target.search('ERBB')
targets = pd.DataFrame.from_dict(target_query)
# print(targets)
selected_target = targets.target_chembl_id[0]
print(selected_target)

# %%
activity = new_client.activity
result = activity.filter(target_chembl_id=selected_target).filter(standard_type="AC50")
df = pd.DataFrame.from_dict(result)
#print(df.head(3))

# %%
df.standard_type.unique()
df.to_csv('data.csv', index=False)

# %%
df2 = df[df.standard_value.notna()]
print(df2)
# %%
bioactivity_class = []
for i in df2.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append('inactive')
    elif float(i) <= 1000:
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")
# %%
mol_cid = []
for i in df2.molecule_chembl_id:
    mol_cid.append(i)

canonical_smiles = []
for i in df2.canonical_smiles:
    canonical_smiles.append(i)

standard_value = []
for i in df2.standard_value:
    standard_value.append(i)

data_tuples = list(zip(mol_cid, canonical_smiles, bioactivity_class, standard_value))
df3 = pd.DataFrame(data_tuples, columns=('mol_cid', 'canonical_smiles', 'bioactivity_class', 'standard_value'))
df3.to_csv('bioactivity_preprocessed_data.csv', index=False)

# %%
