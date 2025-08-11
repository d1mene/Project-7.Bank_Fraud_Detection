import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# random state
rs=42
# чекпоинт данных
def save_data(data, path):
    data.to_csv(path)
# загрузка данных
def open_data(path):
    return pd.read_csv(path, index_col=0)
# функция, сохраняющая список
def save_list(lst, name):
    with open(f'logs/{name}.txt', 'w') as f:
        for i in lst:
            f.write(f'{i}\n')
# функция, открывающий список из логгов          
def open_list(path):
    with open(path, 'r') as f:
        lst = f.read()
    
    lst = lst.split('\n')
    lst.remove('')
    return lst
# функция, сохраняющая модель
def save_model(model, name):
    with open(f'logs/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
# функция, открывающая модель
def open_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model