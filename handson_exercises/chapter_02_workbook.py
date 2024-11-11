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

housing.plot(kind="scatter", 
             x="longitude", 
             y="latitude", 
             grid=True,
             alpha=0.2)
plt.show()

# #### Observations
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

# #### More Correlation Visualization
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


