from os import path

import pandas as pd


class DataReader(object):

    def is_exist(self, filename: str) -> bool:
        return path.exists(filename)


class CSVReader(DataReader):

    def read(self, filename: str) -> pd.DataFrame:
        if self.is_exist(filename):
            return pd.read_csv(filename)
        else:
            raise FileNotFoundError
