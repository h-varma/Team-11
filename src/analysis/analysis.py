import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sbn
from typing import Union

class readIO:
    """ Class for reading in the data.
    Either as numpy array or as pandas Dataframe.

    :Input: Filename and relative path to file.
    :Returns: The data either as Pandas Dataframe or Numpy Array.
    """
    
    def __init__(self, filedir, filename):
        """init function

        :param filedir: relative path
        :type filedir: String
        :param filename: name of the file
        :type filename: String
        """
        self.filedir = filedir
        self.filename = filename

    def read_in_df(self):
        """Read in a pandas csv

        :return: pandas DataFrame
        :rtype: pd.DataFrame
        """
        name = '{}{}'.format(self.filedir, self.filename)
        print('Reading from file {} - pandas'.format(name))
        data = pd.read_csv(name, r'\s+')
        return data

    def read_in_np(self):
        """Read in a numpy array

        :return: numpy array
        :rtype: np.ndarray
        """
        name = '{}{}'.format(self.filedir, self.filename)
        print('Reading from file {} - numpy'.format(name))
        data = np.loadtxt(name, skiprows=1)
        return data

class process:
    """processing methods
    """
    #we need here pd and np as input since our parent class should handle both
    def __init__(self, data:pd.DataFrame):
        """This class is about the processing methods

        :param data: provide a pandas dataframe
        :type data: pd.DataFrame
        """
        self.data = data
        # testing remote-vs-local merge

    def filter_data(self, treshv:float = 1.0e-5):
        """drop columns where the variance is below treshold.
        Only for Pandas DataFrames

        :param treshv: treshold value, defaults to 1.0e-5
        :type treshv: float, optional
        :return: Dataframe without dropped columns
        :rtype: pd.DataFrame
        """
        if type(self.data) != pd.DataFrame:
            print("only implemented for Pandas Dataframe")
            return None
        var = self.data.var(axis=0)
        for column, variance in enumerate(var):
            if var[column] < treshv:
                drop_columns.append(self.data.columns[column])
        return self.data.drop(drop_columns, axis=1)

    def plot_columns(self, idx:int = None):
        """Plot for Pandas DataFrames

        :param idx: index, defaults to None
        :type idx: int, optional
        """
        #maybe change this to a plot function for both types?
        #if type=pandas this
        #elif type=nparray something else
        #or create a method in each child class?
        if type(self.data) == pd.DataFrame:
            if idx == None:
                for col in range(1, len(self.df.columns)):
                    self.df.plot("time", self.df.columns[col])
                    # the replace stuff I added just, because linux isn't happy with </> in the filename name
                    # plt.savefig(str(filename) + str(df.columns[col]).replace("<", "").replace(">", ""))
                    plt.show()
                    plt.close()
            else:
                self.df[self.df.columns[self.idx]].plot()
                #plt.savefig(str(self.filename) + str(self.df.columns[self.idx]).replace("<", "").replace(">", ""))
                plt.show()
                plt.close()
        elif type(self.data) == np.ndarray:
            print("this is an array")
            pass

#for now i just inserted the Methods i thought may be usefull to have in the respective class
class process_statistical(process):
    """Statistical functions

    :param process: inherits class process. Provide data
    :type process: DataFrame or ndarray
    """
    def __init__(self, dataframe):
        """use super is so that child classes that may be using cooperative multiple inheritance 
        will call the correct next parent class function in the Method Resolution Order (MRO)

        :param dataframe: [description]
        :type dataframe: [type]
        """
        super().__init__(dataframe)
    
    def correlation(self):
        """not yet something here

        :return: ?
        :rtype: ?
        """
        # can we somehow identify the "time" column automatically?
        # remove time column and calculate correlation matrix
        wo_time = self.dataframe.drop("time", axis=1)
        corr_mat = wo_time.corr()
        #remove lower triangular
        corr = corr_mat.where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))
        #flatten to vector and remove Nan entries
        corr_vec = corr.unstack().dropna()
        #sort according to absolute value
        return corr_vec.reindex(corr_vec.abs().sort_values(ascending=False).index)
    
    def something_for_L2_Vectornorm(self, indices):
        """?

        :param indices: ?
        :type indices: ?
        :return: ?
        :rtype: ?
        """
        #probably call with one or two list of indices for which columns the norm has to be calculated?
        #maybe another method could generate the lists dependng on whats needed?
        return None
    
#actually you can make this whole calss completely independent since all methods in the parent class
#currently only work for pandas :D
#Maybe we can keep it in case we add a tranformation pandas->numpy or sth like that?
#Or my distribution for the Methods doesn't make too much sense and you have a better idea
class process_numerical(process):
    """Numerical methods

    :param process: data
    :type process: DataFrame or nparray
    """
    def __init__(self, nparray):
        super().__init__(nparray)
    
    def something_for_fft(self):
        return None
    
    def something_for_transforming_to_complex_array(self):
        return None
    
    def something_for_autocorrelation(self):
        return None