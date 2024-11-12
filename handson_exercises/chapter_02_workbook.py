# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: ml_audio_id-venv-9ab27db4d3
#     language: python
#     name: ml_audio_id-venv-9ab27db4d3
# ---

# +
from pathlib import Path
import pandas as pd
from pandas.plotting import scatter_matrix
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


from ipyleaflet import Map, basemaps


# -

# # Get the Data

# +
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
# -

# ## EDA
#
# Quick exploration of the data

housing.head(10)

housing.info()

housing['ocean_proximity'].value_counts()

print(housing.describe().to_markdown())

avg_lat_lon = (float(housing['latitude'].mean()), float(housing['longitude'].mean()),)
Map(center = avg_lat_lon, zoom = 10, min_zoom = 1, max_zoom = 20)

# +
# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
# save_fig("attribute_histogram_plots")  # extra code
plt.show()


# -
# ## Split the training and testing set

def shuffle_and_split_data(data, test_ratio):
    """
    Randomize and split a data into train and test set. 

    DON'T USE THIS. Scikit has a system for doing this to avoid test and train 
    becoming intermingled through randomization in future runs.

    Parameters:
    data (pandas data frame): data to split
    test_ratio (float): fraction of total to reserve as test

    Returns:
    (train_df, test_df)
    """
    shuffled_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indicies[:test_set_size]
    train_indicies = shuffled_indicies[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]


# ### What not to do
#
# Using this custom built function will allow training and testing data to get mixed up on future runs because there's not a good way to ensure that the random permutation won't mix things up if the data set changes, is added to, etc.

train_set, test_set = shuffle_and_split_data(housing, 0.2)
print(f'train_set: {len(train_set)}, test_set: {len(test_set)}')

# ### SciKit has you covered
#
# Do this instead for a stable train/test sets

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# ### Create a new category based on median income

housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                               labels=[1, 2, 3, 4, 5])

housing.head()

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    stratify=housing["income_cat"],
    random_state=42
)

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# +
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
print((compare_props * 100).round(2).to_markdown())
# -
# drop the income-cat column
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)

# ### Save your work...
#
# Write the training sets out to disk for easy loading later without jumping through all the hoops above.

strat_train_set.to_csv("./datasets/strat_train_set-241111.csv", index=False)
strat_test_set.to_csv("./datasets/strat_test_set-241111.csv", index=False)


# And load the sets easily from disk.

strat_train_set = pd.read_csv("./datasets/strat_train_set-241111.csv")
strat_test_set = pd.read_csv("./datasets/strat_test_set-241111.csv")

strat_train_set

# And make a copy of the training set to play with

housing = strat_train_set.copy()

# ## Explore and Visualize the Data to Gain Insights

housing.plot(kind="scatter", 
             x="longitude", 
             y="latitude", 
             grid=True,
             alpha=0.2)
plt.show()

# ### Observations
#
# - Most of the population lives along the coast
# - There's a huge cluster around 38N, 122W (San Francisco?) and 34N, 118W (Los Angeles?)
# - There's also a huge population inland North East of Los Angeles with fairly low value

# +
housing.plot(kind="scatter", 
             x="longitude", 
             y="latitude", 
             grid=True,
             s=housing["population"]/100, # bubble size
             label="population", # key
             c="median_house_value", # color scale
             cmap="jet", # color mapping theme
             colorbar=True, # add color bar key (default is True)
             legend=True, # ???
             sharex=False, # ???
             figsize=(10, 7) # embiggen 
             
            )

plt.show()

# +
# I don't know if I forgot to drop this column, or do some transformation, but the "ocean_proximity"
# value fouls up the correlation calculation because it's not a number.

corr_matrix = housing.drop(columns=["ocean_proximity"]).corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False).to_markdown())
# -

# ### More Correlation Visualization
#
# We're looking for values that might predict the value of a house. The matrix below plots all the values against all the other values. Looking across the median_house_value row, it looks like median_income and possibly total_rooms are helpful indicators.
#

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

housing.plot(kind="scatter",
             x="median_income",
             y="median_house_value",
             alpha=0.1,
             grid=True)
plt.show()

# ### A little more EDA and Experimentation
#
# It might be useful to look at rooms/house and some other values and compute the correlation table again.

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"] 
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.drop(columns=["ocean_proximity"]).corr()
print(corr_matrix["median_house_value"].sort_values(ascending=True).to_markdown())

# ## Prepare the Data for ML
#
# Separate the labels from the training data.

# +
strat_train_set = pd.read_csv("./datasets/strat_train_set-241111.csv")
strat_test_set = pd.read_csv("./datasets/strat_test_set-241111.csv")

# 
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_test_set["median_house_value"].copy()
# -

# Identify null rows
#

null_rows_idx = housing.isnull().any(axis=1)
housing.loc[null_rows_idx].head()

# Use and imputer to apply the median value for each missing feature to make sure that the arrays are the same size for all the data.

# +
# get the median value
median = housing["total_bedrooms"].median()

# fill in the missing values with the imputed value (median total_bedrooms)
housing["total_bedrooms"].fillna(median, inplace=True)

# the above method appears to be deprecated; instead it might make more sense to do:
# housing.loc[:, "total_bedrooms"] = housing["total_bedrooms"].fillna(median)
# -

# is this an alternative to the above? or just another way to do it?
imputer = SimpleImputer(strategy="median")
imputer.strategy


# imputers only work on numerical values; extract just the number type features
housing_num = housing.select_dtypes(include=[np.number])

imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)


X = imputer.transform(housing_num)

imputer.feature_names_in_

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)



housing_tr.loc[null_rows_idx].head()  # not shown in the book

# ### Handle the Text/Categorical Features
#
# It's logical to encode these to numerical values so we can do more mathy things with them

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.value_counts().to_markdown())

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]

ordinal_encoder.categories_

# +
cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
# -

# convert to full array because ??? REASONS ???
housing_cat_1hot.toarray()

cat_encoder.categories_

df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY", "FooBar", "SPAM", "Ham"]})
print(pd.get_dummies(df_test).to_markdown())



# ## Feature Scaling
#
# Scaling features into a range that is appropriate for the ML approach. Typically this means scaling between -1 and 1, or 0 and 1

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

print(housing_num.head(10).to_markdown())

for i in range (0, 10):
    print(housing_num_std_scaled[i])

# ### Custom Transformers
#
# If you don't find just the right transformer, you can write your own class. As long as the class suports `fit` and `transform` methods, it will work. This is great if you really know what you're looking for because you can use pretty much functions or methods you can get your hands on. The one below (borrowed from the text) uses the `rbf_kernel` I can't say that I entirely understand how it works, but I see the potential here.

# ### Pipelines
#
# Rather than performing all the steps one at a time and storing the results in variables, and recombining them manually, SciKit can build a pipeline for you!

# +
import sklearn
sklearn.set_config(display="diagram")

from sklearn.pipeline import Pipeline
    
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
num_pipeline

# +
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
num_pipeline

# +
from pathlib import Path
import pandas as pd
from pandas.plotting import scatter_matrix
import tarfile
import urllib.request

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler #, OrdinalEncoder
from sklearn.model_selection import train_test_split



from sklearn.cluster import KMeans

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"), # (A) impute missing values
        FunctionTransformer(column_ratio, feature_names_out=ratio_name), # (C) Create ratio features
        StandardScaler()) # (F) scale all the values

# load unprocessed data
housing = load_housing_data()

housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                               labels=[1, 2, 3, 4, 5])
# split into train/test split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    stratify=housing["income_cat"],
    random_state=42)

# drop the income-cat column
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing_labels = strat_train_set["median_house_value"].copy()

housing = strat_train_set.drop("median_house_value", axis=1)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), # (A) impute missing values
    OneHotEncoder(handle_unknown="ignore")) # (B) encode categorical data as binary one-hot columns


# (E) transform "long-tail" data into more gaussian (normal) distributions
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"), #(A) impute missing values
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()) # (F) scale all the values
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), # (A) impute missing values
                                     StandardScaler()) # (F) scale all the values
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age
# -

housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape

preprocessing.get_feature_names_out()


