# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:29:10 2023

@author: Strange
"""

import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
PROJECT_ROOT_DIR = "."

def fetch_housing_data(housing_url = HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()
        
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

"""To split data in random pattern """
# def split_training_set(data, test_ratio):
#     shuffled_indices = np.random.seed(42)
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

"""Another way to split trainingh data"""
# from zlib import crc32 #crc32 is a error detecting function that detects the changes in source and target 
# def test_set_checker(identifier, test_ratio):
#     return crc32(np.int64(identifier) & 0Xffffffff < test_ratio * 2**32)

# def split_train_test_by_id(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_checker(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]

# housing_with_id = housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index") # works same as split_training_set

"""This algorithm is from sklearn.model_selection to split data"""
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


"""Creating a income category to use in startified split"""
housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0, 1.5, 3.0, 4.5, 6. , np.inf],
                                labels=[1,2,3,4,5])
# housing["income_cat"].hist()

"""Stratified sampling"""
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
    
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
"""Creating a copy of training set to not mess with the original data"""
housing = strat_train_set.copy()

"""Visualizing graphical data"""
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100,
#               label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"),
#               colorbar=True,)
# plt.legend()

"""Visualizing graphical data over california map image to have a better view"""
# import matplotlib.image as mpimg

'''Downloading California image'''
# images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
# os.makedirs(images_path, exist_ok=True)
# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
# filename = "california.png"
# print("Downloading", filename)
# url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
# urllib.request.urlretrieve(url, os.path.join(images_path, filename))

'''maping it in graph'''
# california_img=mpimg.imread(os.path.join(images_path, filename))
# ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
#                   s=housing['population']/100, label="Population",
#                   c="median_house_value", cmap=plt.get_cmap("jet"),
#                   colorbar=False, alpha=0.4)
# plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
#            cmap=plt.get_cmap("jet"))
# plt.ylabel("Latitude", fontsize=14)
# plt.xlabel("Longitude", fontsize=14)

# prices = housing["median_house_value"]
# tick_values = np.linspace(prices.min(), prices.max(), 11)
# cbar = plt.colorbar(ticks=tick_values/prices.max())
# cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
# cbar.set_label('Median House Value', fontsize=16)

# plt.legend(fontsize=16)
# plt.show()

'''droping ocean_proximity as it is type string and can't convert to float'''
# housing.drop("ocean_proximity", axis=1, inplace=True)

"""Experimenting with Attributes to create some usefull attribures for correlaton"""
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]

# print(housing.info())
'''finding corelation using the function by pandas'''
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


'''Another way to check for correlation (pandas' scatter_matrix function)'''
# from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8)) 

"having a closer look to tehe corelation B/W median income and median house value after observing plotted graphs from above"
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

housing_num = strat_train_set.drop("median_house_value",axis=1)
housing_num.drop("ocean_proximity", axis=1, inplace=True)
housing_labels = strat_train_set["median_house_value"].copy()


"""creating a median to fill n/a values"""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)

# print(imputer.statistics_)


# print(housing.median().values)

x = imputer.transform(housing_num)
'''Training set cleaned and ready'''
housing_tr = pd.DataFrame(x, columns=housing_num.columns)


housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head(10))

"""Handling text and categorical attributes"""
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray())
# print(ordinal_encoder.categories_)


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedroom_per_room = True):
        self.add_bedroom_per_room = add_bedroom_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedroom_per_room:
            bedrooms_per_room = X[:,bedroom_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
        

        
attr_adder = CombinedAttributesAdder(add_bedroom_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])


