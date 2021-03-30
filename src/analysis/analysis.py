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

    def drop_col(self, key: str, name: str):
        """Drops a specific column of a pd dataframe.

        :param key: str for lookup in dict
        :type key: str
        :param name: name of the column to drop
        :type name: str
        """
        df = self.data[key]
        if type(df) != pd.DataFrame:
            print(f"works only for pd Dataframe")
            return None
        self.data[key] = df.drop(name, axis=1)

    def filter_data(self, key: str = "raw_data", treshv: float = 1.0e-5):
        """drop columns where the variance is below treshold.
        Only for Pandas DataFrames

        :param key: key for lookup in dict, default: "raw_data"
        :type key: str
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

    def plot_columns(self, key: str, idx: int = None):
        """Plot for Pandas DataFrames

        :param key: key for lookup in dict
        :type key: str
        :param idx: index, defaults to None
        :type idx: int, optional
        """
        if type(self.data[key]) == pd.DataFrame:
            df = self.data[key]
            if idx is None:
                for col in range(1, len(df.columns)):
                    df.plot("time", df.columns[col])
                    # the replace stuff I added just, because linux isn't happy with </> in the filename name
                    # plt.savefig(str(filename) + str(df.columns[col]).replace("<", "").replace(">", ""))
                    plt.show()
                    plt.close()
            else:
                df[df.columns[idx]].plot()
                # plt.savefig(str(self.filename) + str(self.df.columns[self.idx]).replace("<", "").replace(">", ""))
                plt.show()
                plt.close()
        elif type(self.data[key]) == np.ndarray:
            print("this is an array")
            pass


class ProcessStatistical(Process):
    """
    Class containing methods for statistical analysis.
    It inherits methods for data processing from the Process class.

    :param data: pre-processed data
    :type data: pd.DataFrame or np.ndarray
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        super().__init__(data)

    def correlation(self, key: str):
        """
        Calculates Correlation Matrix of Dataframe, removes the lower triangular.
        The output will be flattened and sorted by absolute value.

        :param key: self.data dictionary value for which the correlation matrix needs to be computed
        :type key: str
        :return: correlation matrix flattened and sorted by absolute value
        :rtype: np.ndarray
        """
        df = self.data[key]
        if type(df) != pd.DataFrame:
            warnings.warn("Evaluation of the correlation works only for pandas Dataframe")
            return None
        # calculate correlation matrix
        corr_mat = df.corr()
        # remove lower triangular
        corr = corr_mat.where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))
        # flatten to vector and remove Nan entries
        corr_vec = corr.unstack().dropna()
        # sort according to absolute value
        return corr_vec.reindex(corr_vec.abs().sort_values(ascending=False).index)

    def distance(self, key: str, idx_list1: list, idx_list2: list):
        """
        Calculates the Euclidean distance between 2 columns of an array.

        :param key: self.data dictionary value for which the distance needs to be computed
        :type key: str
        :param idx_list1: list of column indices
        :type idx_list1: list[int]
        :param idx_list2: list of column indices
        :type idx_list2: list[int]
        :return: L2 distance between the columns idx_list1 and idx_list2 of self.data[key]
        :rtype: np.ndarray
        """
        # maybe another method could generate the lists dependng on whats needed?
        if type(self.data[key]) != np.ndarray:
            warnings.warn("Evaluation of the Euclidean distance works only for numpy array.")
            return None
        if len(idx_list1) != len(idx_list2):
            raise ValueError("You need to pass two lists with the same number of entries")
        array = self.data[key]
        ret = np.zeros(len(idx_list1))
        for i in range(len(idx_list1)):
            ret[i] = np.linalg.norm(array[idx_list1[i]] - array[idx_list2[i]])
        return ret


class ProcessNumerical(Process):
    """
    Class containing methods for numerical analysis.
    It inherits methods for data processing from the Process class.

    :param data: pre-processed data
    :type data: pd.DataFrame or np.ndarray
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        super().__init__(data)

    def discrete_fourier_transform(self, key: str):
        """
        Compute the discrete fourier transform

        :param key: self.data dictionary value for which the discrete fourier transform needs to be computed
        :type key: str
        :return: discrete fourier transform
        :rtype: np.ndarray
        """
        if type(self.data[key]) == pd.DataFrame:
            return np.fft.fft(self.data[key].drop("time", axis=1).values)
        elif type(self.data[key]) == np.ndarray:
            return np.fft.fft(self.data[key])

    def create_complex_matrix(self, key: str):
        """
        Convert the real data matrix containing columns of real and imaginary values into a complex matrix

        :param key: self.data dictionary value where the real and imaginary values are stored
        :type key: str
        """
        complex_matrix = np.empty([self.data[key].shape[0], self.data[key].shape[1] // 2], dtype=np.complex128)
        complex_matrix.real = self.data[key][:, 0::2]
        complex_matrix.imag = self.data[key][:, 1::2]
        self.data["complex_matrix"] = complex_matrix

    def compute_autocorrelation(self, key: str) -> np.ndarray:
        """
        Compute the autocorrelation

        :param key: self.data dictionary value for which the autocorrelation needs to be computed
        :type key: str
        :return: autocorrelation vector
        :rtype: np.ndarray
        """
        total_autocorr = np.zeros(self.data[key].shape[0], dtype=np.complex128)
        for t in range(self.data[key].shape[0]):
            total_autocorr[t] = sum(self.data[key][0] * self.data[key][t])
        return total_autocorr
