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
    res_raw = requests.get(f'{API_URL}/spectra/data?page=1&per_page=50&ids=2,3')

    res = json.loads(json.loads(res_raw.content))

    spectra_data, search_metadata = res['records'], res['metadata']

    print(len(spectra_data))
    print(len(res_raw.content))


def make_metadata_request():
    res_raw = requests.get(f'{API_URL}/spectra/metadata?page=0&per_page=50', params=data)
    res = json.loads(json.loads(res_raw.content))

    spectra_metadata, search_metadata = res['records'], res['metadata']

    df = pd.DataFrame(spectra_metadata)
    print(df.head(5))
    print(search_metadata)
    print(len(res_raw.content))

def make_combo_request():
    res_raw = requests.get(f'{API_URL}/spectra/metadata?page=0&per_page=100', params=data)
    res = json.loads(json.loads(res_raw.content))

    spectra_metadata, search_metadata = res['records'], res['metadata']

    df = pd.DataFrame(spectra_metadata)

    ids = ','.join(df['spectra_id'].astype('str'))

    res_raw = requests.get(f'{API_URL}/spectra/data?page=1&per_page=50&ids={ids}')

    res = json.loads(json.loads(res_raw.content))

    spectra_data, search_metadata = res['records'], res['metadata']

    print(f'Retrieved {len(spectra_data)} spectra, total bytes {len(res_raw.content)}')


print(timeit.timeit(make_combo_request, number=10))
