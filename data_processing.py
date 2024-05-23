import pandas as pd
from chembl_webresource_client import new_client

target = new_client.target
target_query = target.search('ERBB')
targets = pd.DataFrame.from_dict(target_query)
