import requests
import json
import timeit
import pandas as pd
from io import StringIO

ne_data = {}
ne_be_data = {'element': ['Ne', 'Ba']}

data = ne_data


def make_spectra_request():
    r = requests.get(r'http://67.205.178.106/spectra', data=json.dumps(data))
    with open('readTest/read_test.zip', 'wb') as fo:
        fo.write(r.content)


def make_metadata_request():
    r = requests.get(r'http://67.205.178.106/spectra/metadata', data=json.dumps(data))
    metadata = json.loads(r.content)
    print(metadata)
    df = pd.read_json(StringIO(metadata), orient='records')
    print(df.columns)


print(timeit.timeit(make_metadata_request, number=10))
