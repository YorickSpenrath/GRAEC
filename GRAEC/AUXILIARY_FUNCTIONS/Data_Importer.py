"""
Used to import data. I created this way before I knew about the magic of pandas. Fixing this to pandas is a TODO
There is however no priority whatsoever
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math

from GRAEC.Functions import Filefunctions


class DataImporter:

    def __init__(self, fn_data):

        # get header data
        with open(fn_data, 'r+') as rf:
            header = np.array(rf.readline()[:-1].split(';'))
            numerical = np.array(rf.readline()[:-1].split(';')) == 'TRUE'
            categorical = np.array(rf.readline()[:-1].split(';')) == 'TRUE'
            data_type = np.array(rf.readline()[:-1].split(';'))

            self.X_mask = data_type == 'X'
            self.y_mask = data_type == 'Y'
            self.split_mask = data_type == 'SPLIT'
            self.ID_mask = data_type == 'ID'
            self.categorical_X = categorical[self.X_mask]
            self.NUMERICAL_X = numerical[self.X_mask]
            self.header_X = header[self.X_mask]
            self.header_y = header[self.y_mask]

            x = []
            y = []
            split = []
            ids = []

            for line in rf.readlines():
                row = line[:-1].split(";")
                if not ('null' in row or 'NULL' in row or '' in row):
                    row = np.array(row)
                    x += [row[self.X_mask]]
                    y += [row[self.y_mask]]
                    split += [row[self.split_mask]]
                    ids += [row[self.ID_mask]]

        x = np.array(x).reshape(-1, sum(self.X_mask))
        y = np.array(y).reshape(-1, sum(self.y_mask))
        split = np.array(split).reshape(-1, sum(self.split_mask))
        ids = np.array(ids).reshape(-1, sum(self.ID_mask))

        # Label encoder
        self.LE_X = dict()
        for i in np.nditer(np.where(np.logical_not(self.NUMERICAL_X))):
            le = LabelEncoder()
            x[:, i] = le.fit_transform(x[:, i])
            self.LE_X[str(i)] = le
        self.accepted_fraction = 1

        # Categorical Encoder
        one_hot_encoder = OneHotEncoder(categorical_features=self.categorical_X)
        one_hot_encoder.fit(x)
        self.OHE_X = one_hot_encoder
        x = self.OHE_X.transform(x)

        self.X = x
        self.y = y
        self.split_values = np.array([float(i) for i in split])
        self.IDS = np.array([i[0] for i in ids])

    def get_data(self, fn_subset_ids=None, return_split_values=False, return_identifiers=False):

        if fn_subset_ids is None:
            ret = (self.X.toarray(), self.y)
            ret += (self.split_values,) if return_split_values else ()
            ret += (self.IDS,) if return_identifiers else ()
        else:
            if not Filefunctions.exists(fn_subset_ids):
                raise Exception('File does not exist:\n{}'.format(fn_subset_ids))
            with open(fn_subset_ids, 'r') as rf:
                keep_ids = [line[:-1] for line in rf.readlines()]
            keep_idx = [i for i in range(len(self.IDS)) if self.IDS[i] in keep_ids]
            ret = (self.X.toarray()[keep_idx], self.y[keep_idx])
            ret += (self.split_values[keep_idx],) if return_split_values else ()
            ret += (self.IDS[keep_idx],) if return_identifiers else ()

        return ret

    def split_data(self, interval, fn_subset_ids=None, return_split_values=False, return_identifiers=False):
        if fn_subset_ids is not None:
            if not Filefunctions.exists(fn_subset_ids):
                raise Exception('File does not exist:\n{}'.format(fn_subset_ids))

        keep_idx = None
        if fn_subset_ids is not None:
            with open(fn_subset_ids, 'r') as rf:
                keep_ids = [line[:-1] for line in rf.readlines()]
            keep_idx = [i for i in range(len(self.IDS)) if self.IDS[i] in keep_ids]

        index_list = dict()
        for (i, s) in enumerate(self.split_values):
            if keep_idx is not None and i not in keep_idx:
                continue
            group = math.floor(float(s) / interval)
            index_list.setdefault(group, []).append(i)

        return_x = dict()
        return_y = dict()
        return_split = dict()
        return_ids = dict()

        for (group, indices) in index_list.items():
            return_x[group] = self.X.toarray()[indices]
            return_y[group] = self.y[indices]
            if return_split_values:
                return_split[group] = self.split_values[indices]
            if return_identifiers:
                return_ids[group] = self.IDS[indices]

        ret = (return_x, return_y,)
        ret += ((return_split,) if return_split_values else ())
        ret += ((return_ids,) if return_identifiers else ())
        return ret

    def get_split_values(self):
        return np.array([float(i) for i in self.split_values])
