import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sbn
from typing import Union
import warnings
import copy


class ReadIO:
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


class Process:
    """processing methods
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        """This class is about the processing methods

        :param data: provide a pandas dataframe or numpy array
        :type data: pd.DataFrame or np.ndarray
        """
        self.data = {"raw_data": data}

    def filter_data(self, key, treshv: float = 1.0e-5):
        """drop columns where the variance is below treshold.
        Only for Pandas DataFrames

        :param treshv: treshold value, defaults to 1.0e-5
        :type treshv: float, optional
        :return: Dataframe without dropped columns
        :rtype: pd.DataFrame
        """
        if type(self.data[key]) != pd.DataFrame:
            warnings.warn("Filter data is only implemented for Pandas Dataframe")
            return None
        df = self.data[key]
        var = df.var(axis=0)
        drop_columns = []
        for column, variance in enumerate(var):
            if var[column] < treshv:
                drop_columns.append(df.columns[column])
        self.data["filtered_data"] = df.drop(drop_columns, axis=1)

    def plot_columns(self, key, idx: int = None):
        """Plot for Pandas DataFrames

        :param idx: index, defaults to None
        :type idx: int, optional
        """
        if type(self.data) == pd.DataFrame:
            if idx is None:
                for col in range(1, len(self.data.columns)):
                    self.data.plot("time", self.data.columns[col])
                    # the replace stuff I added just, because linux isn't happy with </> in the filename name
                    # plt.savefig(str(filename) + str(df.columns[col]).replace("<", "").replace(">", ""))
                    plt.show()
                    plt.close()
            else:
                self.data[self.data.columns[idx]].plot()
                # plt.savefig(str(self.filename) + str(self.df.columns[self.idx]).replace("<", "").replace(">", ""))
                plt.show()
                plt.close()
        elif type(self.data) == np.ndarray:
            print("this is an array")
            pass


class ProcessStatistical(Process):
    """Statistical functions

    :param Process: inherits class process. Provide data
    :type Process: DataFrame or ndarray
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        """use super is so that child classes that may be using cooperative multiple inheritance 
        will call the correct next parent class function in the Method Resolution Order (MRO)

        :param dataframe: [description]
        :type dataframe: [type]
        """
        super().__init__(data)

    def correlation(self):
        """calculates Correlation Matrix of Dataframe, removes the lower triangular.
        The output will be flattened and sorted by absolute value.

        :return: ?
        :rtype: ?
        """
        # calculate correlation matrix
        corr_mat = self.data.corr()
        # remove lower triangular
        corr = corr_mat.where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))
        # flatten to vector and remove Nan entries
        corr_vec = corr.unstack().dropna()
        # sort according to absolute value
        return corr_vec.reindex(corr_vec.abs().sort_values(ascending=False).index)

    def distance(self, idx_list1, idx_list2):
        """calculates the euclidean distance between 2 columns of an array.
        The column indices are provided as lists. Returns the distance Norm(ar[idx_list1[i]]-ar[idx_list2[i]]).

        :param idx_list1: list of indices
        :type idx_list1: list[int]
        :param idx_list2: list of indices
        :type idx_list2: list[int]
        :return: Vector with L2 Norms
        :rtype: np.ndarray
        """
        # maybe another method could generate the lists dependng on whats needed?
        if type(self.data) != np.ndarray:
            print("works only for numpy array")
            return None
        if len(idx_list1) != len(idx_list2):
            print("you need to pass to list with the same number of entries")
            return None
        ret = np.zeros(len(idx_list1))
        for i in range(len(idx_list1)):
            ret[i] = np.linalg.norm(self.data[idx_list1[i]] - self.data[idx_list2[i]])
        return ret


class ProcessNumerical(Process):
    """
    Class containing methods for numerical analysis.
    It inherits methods for data processing from the Process class.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        :param data: pre-processed data
        :type data: pd.DataFrame or np.ndarray
        """

        super().__init__(data)

    def discrete_fourier_transform(self, data: Union[(pd.DataFrame, np.ndarray)] = None):
        """
        Compute the discrete fourier transform

        :param data: input data to apply fourier transform on. if None, self.data is used.
        :type data: pd.DataFrame or np.ndarray
        :return: discrete fourier transform
        :rtype: np.ndarray
        """
        if data is None:
            data = copy.deepcopy(self.data)
        if type(data) == pd.DataFrame:
            return np.fft.fft(data.drop("time", axis=1).values)
        elif type(data) == np.ndarray:
            return np.fft.fft(data)

    def create_complex_matrix(self):
        """
        Convert the real data matrix containing columns of real and imaginary values into a complex matrix
        """
        complex_matrix = np.empty([self.data.shape[0], self.data.shape[1] // 2], dtype=np.complex128)
        complex_matrix.real = self.data[:, 0::2]
        complex_matrix.imag = self.data[:, 1::2]
        self.data = complex_matrix

    def compute_autocorrelation(self, data: np.ndarray = None):
        """
        Compute the autocorrelation

        :param data: input data to compute the autocorrelation for. if None, self.data is used.
        :type data: np.ndarray
        :return: autocorrelation
        :rtype: np.ndarray
        """
        if data is None:
            data = copy.deepcopy(self.data)
        total_autocorr = np.zeros(data.shape[0], dtype=np.complex128)
        for t in range(data.shape[0]):
            total_autocorr[t] = sum(data[0] * data[t])
        return total_autocorr
