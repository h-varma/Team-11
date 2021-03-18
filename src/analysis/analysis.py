import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn

class readIO:
    def __init__(self, filedir, filename):
        self.filedir = filedir
        self.filename = filename

    def read_in_df(self):
        name = '{}{}'.format(self.filedir, self.filename)
        print('Reading from file {} - pandas'.format(name))
        data = pd.read_csv(name, r'\s+')
        return data

    def read_in_np(self):
        name = '{}{}'.format(self.filedir, self.filename)
        print('Reading from file {} - numpy'.format(name))
        data = np.loadtxt(name, skiprows=1)
        return data

class data_preparation:
    pass

class statistical_methods:
    pass

class numerical_methods:
    pass
