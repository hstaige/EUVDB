import requests
import json
import timeit

ne_data = {'element': ['Ne', 'Ba'], }
ne_be_data = {'element': ['Ne', 'Ba']}

data = ne_data


def make_request():
    r = requests.put(r'http://127.0.0.1:8000/spectra', data=json.dumps(data))
    with open('readTest/read_test.zip', 'wb') as fo:
        fo.write(r.content)


print(timeit.timeit(make_request, number=10))
