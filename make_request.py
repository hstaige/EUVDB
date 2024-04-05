import requests
import json

ne_data = {'element': ['Ne'], 'lower_date_taken': '2020/01/01'}
ne_be_data = {'element': ['Ne', 'Ba']}

data = ne_data

r = requests.put(r'http://127.0.0.1:8000/spectra', data=json.dumps(data))
with open('readTest/read_test.zip', 'wb') as fo:
    fo.write(r.content)
