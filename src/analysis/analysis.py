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
    def __init__(self, data):
        self.data = data
        # testing remote-vs-local merge

    def filter_data(self, treshv = 1.0e-5):
        if type(self.data) != pd.DataFrame:
            print("only implemented for Pandas Dataframe")
            return None
        var = self.data.var(axis=0)
        for column, variance in enumerate(var):
            if var[column] < treshv:
                drop_columns.append(self.df.columns[column])
        return self.df.drop(drop_columns, axis=1)

    def plot_columns(self, idx = None):
        #maybe change this to a plot function for both types?
        #if type=pandas this
        #elif type=nparray something else
        #or create a method in each child class?
        if type(self.data) != pd.DataFrame:
            print("only implemented for Pandas Dataframe")
            return None
        if idx == None:
            for col in range(1, len(self.df.columns)):
                self.df.plot("time", self.df.columns[col])
                # the replace stuff I added just, because linux isn't happy with </> in the filename name
                # plt.savefig(str(filename) + str(df.columns[col]).replace("<", "").replace(">", ""))
                plt.show()
                plt.close()
        else:
            self.df[self.df.columns[self.idx]].plot()
            plt.savefig(str(self.filename) + str(self.df.columns[self.idx]).replace("<", "").replace(">", ""))

#for now i just inserted the Methods i thought may be usefull to have in the respective class
class process_statistical(process):
    def __init__(self, dataframe):
        super().__init__(dataframe)
    
    def something_for_correlation_matrix(self):
        return None
    
    def something_for_L2_Vectornorm(self, indices):
        #probably call with one or two list of indices for which columns the norm has to be calculated?
        #maybe another method could generate the lists dependng on whats needed?
        return None
    
#actually you can make this whole calss completely independent since all methods in the parent class
#currently only work for pandas :D
#Maybe we can keep it in case we add a tranformation pandas->numpy or sth like that?
#Or my distribution for the Methods doesn't make too much sense and you have a better idea
class process_numerical(process):
    def __init__(self, nparray):
        super().__init__(nparray)
    
    def something_for_fft(self):
        return None
    
    def something_for_transforming_to_complex_array(self):
        return None
    
    def something_for_autocorrelation(self):
        return None