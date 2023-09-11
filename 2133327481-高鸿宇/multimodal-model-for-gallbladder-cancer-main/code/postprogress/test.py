import pathlib

import tensorflow as tf

from data import Data
from model import Model


def test_read_csv():
    path = 'data\data.csv'
    csv = Data.read_csv(path)
    print(csv)
    return csv


def test_get_nii_path():
    path = 'data\data.csv'
    path = Data.get_nii_path(path)
    print(path)
    return path


def test_read_nii_data():
    path = test_get_nii_path()
    data = Data.read_nii_data(pathlib.Path(path), 'Bao Wan Min')
    print(data)
    return data


def test_convert_to_binary():
    data = test_read_nii_data()
    data = Data.convert_to_binary(data)
    print(data)
    return data

def test_get_dataset():
    # 启用某些功能
    tf.config.run_functions_eagerly(True)
    path = 'data\data.csv'
    t, v = Data.get_dataset(path, 12)
    print(t, v)
    return t,v

def test_get_model_path():
    path = Model.get_model_path()
    print(path)
    return path

def test_restore_model():
    model = Model.restore_model()
    print(model)
    return model