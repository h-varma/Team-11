import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn

class readIO:
    """ Class for reading in the data.
    Either as numpy array or as pandas Dataframe.

    :Input: Filename and relative path to file.
    :Returns: The data either as Pandas Dataframe or Numpy Array."""
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

class process:
    def __init__(self, data, treshv):
        self.data = data
        self.treshv = treshv

    def filter_data(self):
        if type(self.data) != pd.DataFrame:
            print("only implemented for Pandas Dataframe")
            return None
        var = self.data.var(axis=0)
        drop_columns = []
        for column, variance in enumerate(var):
            if var[column] < self.treshv:
                drop_columns.append(self.df.columns[column])
        return self.df.drop(drop_columns, axis=1)

    def plot_columns(self):
        if self.idx == None:
            for col in range(1, len(self.df.columns)):
                self.df.plot("time", self.df.columns[col])
                # the replace stuff I added just, because linux isn't happy with </> in the filename name
                # plt.savefig(str(filename) + str(df.columns[col]).replace("<", "").replace(">", ""))
                plt.show()
                plt.close()
        else:
            self.df[self.df.columns[self.idx]].plot()
            plt.savefig(str(self.filename) + str(self.df.columns[self.idx]).replace("<", "").replace(">", ""))

test = np.linspace(0,8,9)
process(test, 2).filter_data()

class process_numerical(process):
    pass

class process_statistical(process):
    pass