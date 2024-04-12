from fastapi import FastAPI, HTTPException

import numpy as np
import h5py
import pandas as pd

from types import UnionType
from typing import get_origin, get_args, Any, Literal
from pydantic import BaseModel
from numbers import Number

import re
import json
import os
from datetime import datetime
from collections import defaultdict
from uuid import uuid4
from io import StringIO
# from pkg_resources import resource_filename
# noinspection PyUnresolvedReferences
# from euv_fitting.calibrate.utils import SpeReader # Used to create database, not neccesary for API
from zipfile import ZipFile

app = FastAPI()
CHUNK_SIZE = 1024 * 1024  # 1 MB of data
H5_FILE_LOC = "newdb.h5"
META_FILE_LOC = "EUV_meta.pkl"


# noinspection PyPep8Naming
class Spectra_Metadata(BaseModel):
    element: str | None
    beam_energy_eV: float | None
    beam_current_mA: float | None
    gain: float | None = None

    date_taken: str | None = None
    time_taken: str | None = None
    date_added: str | None = None
    time_added: str | None = None

    added_by: str | None = None
    comments: list[str] | None = None
    calibration_links: list[str] | None = None
    preferred_calibration: str | None = None
    line_id_link: str | None = None
    paper_links: list[str] | None = None
    lab_book_links: list[str] | None = None


def check_parameterized_generic(field_type):
    try:  # issubclass throws error if field_type includes list[any_type]
        issubclass(Any, field_type)
        return False
    except TypeError:
        return True


def check_numeric(field_type):
    if get_origin(field_type) == UnionType:
        return any(issubclass(ft, Number) for ft in get_args(field_type))
    return issubclass(field_type, Number)


def make_search_class(metadata_class, needs_bounds: list | None = None):
    # Make Pydantic class for FastAPI searching from dataset model class

    cls_dict = {"__annotations__": {}}
    for key, field_type in metadata_class.__annotations__.items():

        cls_dict['__annotations__'][key] = field_type | None
        cls_dict[key] = None

        if check_parameterized_generic(field_type):
            continue

        # Add lower and upper bounds for search
        if check_numeric(field_type) or (needs_bounds and key in needs_bounds):
            for prefix in ('lower_', 'upper_'):
                new_key = prefix + key
                cls_dict['__annotations__'][new_key] = field_type | None
                cls_dict[new_key] = None

        if issubclass(str, field_type):
            cls_dict['__annotations__'][key] = list[str] | None
            cls_dict[key] = None

    cls = type("Search_" + type(metadata_class).__name__, (BaseModel,), cls_dict)
    cls.validate_submission = validate_submission
    return cls


def validate_submission(self):
    for attr, model_field in self.__fields__.items():
        attr_type = model_field.annotation
        attr_val = getattr(self, attr)

        if attr_val is None:
            continue

        if check_parameterized_generic(attr_type):
            continue

        if check_numeric(attr_type):
            if 'lower_' in attr or 'upper_' in attr:
                base_attr = attr[6:]
                if getattr(self, base_attr) is not None:
                    raise HTTPException(status_code=415, detail=f'Only one of {attr} and {base_attr} can be query '
                                                                f'parameters.')

    return True


# noinspection PyPep8Naming
class New_EUVH5_Handler:

    curr_id: int = 0

    def __init__(self, in_file=None, meta_file=None, h5_file=None, spe_folder=None):
        self.in_file = in_file  # if in_file else resource_filename('euv_fitting.euvh5', 'EUV_data.xlsx')
        self.h5_file = h5_file  # if h5_file else resource_filename('euv_fitting.euvh5', 'EUV.h5')
        self.meta_file = meta_file  # if meta_file else resource_filename('euv_fitting.euvh5', 'EUV_meta.pkl')
        self.spe_folder = spe_folder  # if spe_folder else resource_filename('euv_fitting.euvh5', 'EUV_CCD2')

        self.RESULTS_DIR = 'results/'
        self.DATE_FORMAT = "%Y/%m/%d"
        self.TIME_FORMAT = "%H:%M:%S"

        self.excel_cols_to_add = ['Z', 'Element', 'File name', 'Beam E eV', 'Beam C mA', 'Dump s']

        self.regex_recipes = {
            'beam_energy_eV': [(r'(\d+p\d+)[kK][eE]?[Vv]', 1000), (r'E(\d+)', 1), (r'(\d+p\d+)[eE][vV]', 1)],
            'beam_current_mA': [(r'(\d+p?\d?)mA', 1)],
            'dump_s': [(r'(\d+)_?[sS][cC]ook', 1)]}

        self.aliases = {'Beam E eV': 'beam_energy_eV',
                        'Beam C mA': 'beam_current_mA',
                        'rawdate': 'date_taken',
                        'rawtime': 'time_taken',
                        'Dump s': 'dump_s'}

        self.meta_dict = defaultdict(list)

    def create_database(self):

        df = pd.read_excel(self.in_file)
        runs = df['Run'].unique()

        f = h5py.File(self.h5_file, 'w')

        try:
            self.curr_id = 0
            bad_links = 0
            for run in runs:
                # create run group
                run_df = df[df['Run'] == run]
                g = f.create_group(run)
                for idx, row in run_df.T.items():
                    if not pd.isna(row['Subfolder']):
                        spe_path = self.spe_folder + '/' + run + '/' + row['Subfolder']
                    else:
                        spe_path = self.spe_folder + '/' + run
                    try:
                        spe_path = spe_path + '/' + row['File name']
                    except Exception as e:
                        print(repr(e))
                        bad_links += 1
                        print(spe_path, row['File name'])
                        continue

                    try:
                        S = SpeReader(spe_path)
                    except Exception as e:
                        print(repr(e))
                        bad_links += 1
                        print(spe_path, )
                        continue

                    try:
                        arr = S.load_img()
                        ds = g.create_dataset(row['File name'].replace('.spe', ''), data=arr, compression='gzip')
                        ds.attrs['run'] = run
                    except ValueError:
                        print(S.get_size(), spe_path)
                        bad_links += 1
                        continue

                    self.write_attrs(ds, S.metadata, row)

            df = pd.DataFrame(self.meta_dict)
            df.to_pickle(self.meta_file)

            print(f'bad_links = {bad_links}')
        except Exception as e:
            print(repr(e))
        finally:
            f.close()

    def add_auto_attrs(self, dataset):
        now = datetime.now()
        dataset.attrs['date_added'] = now.strftime(self.DATE_FORMAT)
        dataset.attrs['time_added'] = now.strftime(self.TIME_FORMAT)
        dataset.attrs['uuid'] = str(uuid4().bytes)
        dataset.attrs['spectra_id'] = self.curr_id
        dataset.attrs['path'] = dataset.name
        self.curr_id += 1
        return dataset

    def write_meta_dict(self, dataset):
        for key, value in dataset.attrs.items():
            if not self.meta_dict[key]:
                print('new key', key)
            self.meta_dict[key].append(value)

        for key in self.meta_dict:  # check if this ds is missing a value
            if key not in dataset.attrs:
                self.meta_dict[key].append(None)

    def write_attrs(self, dataset, meta_data, row):
        for key, value in meta_data.items():
            key = self.aliases.get(key, key.lower().replace(' ', '_'))
            if key == 'time_taken' and value:
                value = ':'.join(value[i:i + 2] for i in range(0, len(value), 2))
            if key == 'date_taken' and value:
                value = datetime.strptime(value, "%d%b%Y").strftime(self.DATE_FORMAT)
            if key == 'file_name' and value:
                value = value.replace('.SPE', '')
            try:
                dataset.attrs[key] = value
            except ValueError:
                print('Had issues adding', key, value, row['File name'])

        for key in self.excel_cols_to_add:
            value = row[key]
            key = self.aliases.get(key, key.lower().replace(' ', '_'))

            regex_success = False
            if isinstance(value, float):
                if pd.isna(value) and key in self.regex_recipes.keys():
                    for regex, scale in self.regex_recipes[key]:
                        match = re.search(regex, dataset.attrs['file_name'])
                        if match:
                            match_str = match.groups()[0]
                            match_str = match_str.replace('p', '.')
                            dataset.attrs[key] = float(match_str) * scale
                            regex_success = True
                            break
            if not regex_success:
                if key not in ['element', 'file_name']:
                    dataset.attrs[key] = float(value)
                else:
                    dataset.attrs[key] = value

        self.add_auto_attrs(dataset)
        self.write_meta_dict(dataset)

    def search(self, query_dict, return_type: Literal["metadata", "spectra"] = "metadata", per_page=0, page_number=0):
        # filters df until out of query_dict, then retrieves files from EUV.h5
        df = pd.read_pickle(self.meta_file)

        for key, search_value in query_dict.items():
            if 'date_' in key:
                suffix = key.split('date_')[1]
                time = query_dict.get('time_' + suffix, '00:00:00')
                search_value = datetime.strptime(search_value + time, self.DATE_FORMAT + self.TIME_FORMAT)

                if 'datetime_' + suffix not in df:
                    def str_to_datetime(x):
                        return datetime.strptime(x, self.DATE_FORMAT + self.TIME_FORMAT)

                    df = df[df['date_' + suffix].notna() & df['time_' + suffix].notna()]

                    df['datetime_' + suffix] = (df['date_' + suffix] + df['time_' + suffix]).apply(str_to_datetime)

                key = key.replace('date', 'datetime')

            if 'lower_' in key:
                suffix = key.split('lower_')[1]
                df = df[df[suffix] >= search_value]

            elif 'upper_' in key:
                suffix = key.split('upper_')[1]
                df = df[df[suffix] <= search_value]

            else:
                search_value = set(search_value) if type(search_value) is list else {search_value}
                if df[key].dtype == list:
                    df = df[df[key].apply(lambda x: bool(search_value & {x}))]  # check if search_value and row overlap
                else:
                    df = df[df[key].isin(search_value)]

        if per_page == 0:
            per_page = len(df) - 1
            page_number = 0
            low_index, high_index = 0, len(df) - 1
        else:
            low_index, high_index = per_page * page_number, min(len(df) - 1, per_page * (page_number+1))

        search_metadata = {"page": page_number, "per_page": per_page,
                           "page_count": high_index - low_index, "total_count": len(df)}

        df = df.iloc[low_index:high_index]

        if return_type == 'metadata':
            return df, search_metadata
        elif return_type == 'spectra':
            f = h5py.File(self.h5_file, 'r')
            return f, [f[path] for path in df['path']], search_metadata


# def create_datasets_archive(datasets, filename='write_test.zip'):
#     zip_archive_path = os.path.join('writeTest', filename)
#     with open(zip_archive_path, 'wb') as fo:
#         with ZipFile(fo, "w") as zip_archive:
#             for dataset in datasets:
#                 write_dataset_to_archive(dataset, zip_archive)
#
#
# def write_dataset_to_archive(dataset, zip_archive):
#
#     filepath = dataset.name.replace('/', '_') + '.txt'
#     attr_str = json.dumps({key: value for key, value in dataset.attrs.items()}, default=np_encoder)
#     s = StringIO()
#     np.savetxt(s, filter_cosmic_rays(dataset[...]), newline=',', fmt='%8d')
#
#     zip_archive.writestr(filepath, attr_str + '\n' + '-' * 32 + '\n' + s.getvalue())


def filter_cosmic_rays(data, filterval=5, combine=True):
    # Vectorized form of CosmicRayFilter
    # In code below, loo stands for Leave One Out
    data = data.copy()
    nr_frames, camera_size = data.shape

    if nr_frames == 1:
        return data

    sqrts = np.sqrt(data)
    noises = np.expand_dims(np.sum(sqrts, axis=0), axis=0)
    noises = np.repeat(noises, axis=0, repeats=nr_frames)
    loo_noises = (noises - sqrts) / (nr_frames - 1)

    means = np.expand_dims(np.sum(data, axis=0), axis=0)
    means = np.repeat(means, axis=0, repeats=nr_frames)
    loo_means = (means - data) / (nr_frames - 1)
    residuals = loo_means - data

    mask = residuals >= (filterval * loo_noises)
    data[mask] = loo_means[mask]

    if combine:
        return data.sum(axis=0)
    return data


NEUV = New_EUVH5_Handler(h5_file=H5_FILE_LOC,
                         meta_file=META_FILE_LOC)

Search_Spectra_Metadata = make_search_class(Spectra_Metadata, needs_bounds=['date_taken', 'date_added'])


def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()


@app.get("/hello")
def read_root(test: int = 1, page_number: int = 0):
    print(test, page_number)
    return {"Hello": "World"}


@app.get("/spectra/metadata/")
def get_metadata(search_spectra_metadata: Search_Spectra_Metadata, page=0, per_page=0):
    ssm = search_spectra_metadata
    ssm.validate_submission()

    spectra_metadata, search_metadata = NEUV.search({key: value for key, value in vars(ssm).items() if value is not None},
                                                    return_type="metadata", page_number=page, per_page=per_page)

    response = dict()
    response['records'] = spectra_metadata.to_dict(orient='list')
    response['metadata'] = search_metadata

    return json.dumps(response, default=np_encoder)


@app.get("/spectra/data")
def get_spectra(search_spectra_metadata: Search_Spectra_Metadata, page: int = 0, per_page: int = 0):
    ssm = search_spectra_metadata
    ssm.validate_submission()

    f, datasets, search_metadata = NEUV.search({key: value for key, value in vars(ssm).items() if value is not None},
                                               return_type="spectra", page_number=page, per_page=per_page)

    response = dict()
    response['records'] = {ds.attrs['path']: {'data': ds[...], **ds.attrs} for ds in datasets}
    response['metadata'] = search_metadata

    f.close()

    return json.dumps(response, default=np_encoder)
