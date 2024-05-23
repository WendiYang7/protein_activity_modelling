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
print(df.head(3))
# %%
