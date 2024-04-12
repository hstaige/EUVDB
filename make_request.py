import requests
import json
import timeit
import pandas as pd
from io import StringIO

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 180)

API_URL = "http://127.0.0.1:8000"
# API_URL = "http://67.205.178.106"

ne_data = {}
ne_be_data = {'element': ['Ne', 'Ba']}

data = ne_be_data


def make_spectra_request():
    res = requests.get(f'{API_URL}/spectra/data/?page=0&per_page=50', data=json.dumps(data))

    res = json.loads(json.loads(res.content))

    spectra_data, search_metadata = res['records'], res['metadata']

    print(len(spectra_data))


def make_metadata_request():
    res = requests.get(f'{API_URL}/spectra/metadata', data=json.dumps(data))
    res = json.loads(json.loads(res.content))

    spectra_metadata, search_metadata = res['records'], res['metadata']

    df = pd.DataFrame(spectra_metadata)
    print(df.head(5))
    print(search_metadata)


print(timeit.timeit(make_spectra_request, number=10))
