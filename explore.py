import wrangle_zillow

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats

from itertools import combinations

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


def temp_data_scaler(df, columns_to_scale):
    '''
    This function takes in train, validate, test subsets of the cleaned zillow dataset and using the train subset creates a min_max 
    scaler. It thens scales the subsets and returns the train, validate, test subsets as scaled versions of the initial data.

    Parameters:  train, validate, test - split subsets from of the cleaned zillow dataframe
                columns_to_scale - a list of column names to scale
    Return: scaled_train, scaled_validate, scaled_test - dataframe with scaled versions of the initial unscaled dataframes 
    '''
    df_scaled = df.copy()
    
    scaler = MinMaxScaler()
    
    df_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(df[columns_to_scale]), 
                                                  columns=df[columns_to_scale].columns.values).set_index([df.index.values])
    return df_scaled

def plot_variable_pairs(df, to_drop = None, hue_selection=None):
    '''
    This function takes in a dataframe and plots all possible numerical data pairs in scatterplots with a regression 
    line. The function only plots unique combinations of pairs, not permutations, e.g. only prints column a by 
    column b, but leaves out column b by column a. This reduces clutter and runtime.
    
    Paramters: df - A dataframe with numerical columns
               to_drop - a list of columns to drop or not include in the plots
    Returns: This function returns nothing; it merely plots out the scatterplots
    '''
    
    if to_drop == None:
        
        columns_to_plot = df.select_dtypes(include = 'number').columns
        plot_tuples = []
        for i in combinations(columns_to_plot,2):
            plot_tuples.append(i)
        if hue_selection == None:
            for i in plot_tuples:
                sns.lmplot(x = i[0], y = i[1], data = df, line_kws={'color': '#FF5E13', 'linewidth': 3},  height=5, aspect=1.5)
                plt.plot()
                plt.show()
        else:
            plt.figure(figsize=(5,5))
            for i in plot_tuples:
                sns.scatterplot(x = i[0], y = i[1], data = df, hue= hue_selection)
                plt.plot()
                plt.show()
            
    else:
        columns_to_plot = df.select_dtypes(include = 'number').columns.drop(to_drop)
        plot_tuples = []
        for i in combinations(columns_to_plot,2):
            plot_tuples.append(i)
        if hue_selection == None:
            for i in plot_tuples:
                sns.lmplot(x = i[0], y = i[1], data = df, line_kws={'color': '#FF5E13', 'linewidth': 3},  height=5, aspect=1.5)
                plt.plot()
                plt.show()
        else:
            plt.figure(figsize=(5,5))
            for i in plot_tuples:
                sns.scatterplot(x = i[0], y = i[1], data = df, hue= hue_selection)
                plt.plot()
                plt.show()


def plot_categorical_and_continuous_vars(df, continuous, categorical):
    '''
    This function takes in a dataframe, a list of continuous variables, and a list of categorical variables and does 
    3 plots for each unique combination of categorical and continuous variable.
    
    Parameters: df - a dataframe consisting of continuous and categorical columns
    '''
    plot_list = []
    for cat in categorical:
        for cont in continuous:
            plot_list.append([cat, cont])
    
    for i in plot_list:
        plt.figure(figsize=(18, 5))
        plt.subplot(131)
        sns.boxplot(x=i[0], y=i[1], data=df)
        plt.subplot(132)
        sns.stripplot(x=i[0], y=i[1], data=df)
        plt.subplot(133)
        sns.violinplot(x=i[0], y=i[1], data=df)
#         sns.barplot(x=i[0], y=i[1], data=df)
        plt.show()

# Writing a function to determine the sale time in 4 month blocks.

def four_month_split(df):
    '''
    This function takes in a dataframe, reads a transactiondate column and returns a month group category.

    Parameters: df - a dataframe with a transactiondate column in the format "YEAR-MONTH-DAY"

    Returns: a month group category of either Jan-Apr, May-Aug, Sept-Dec
    '''
    if (df['transactiondate'] >= pd.to_datetime('2017-01-01')) and (df['transactiondate'] <  pd.to_datetime('2017-05-01')):
        return 'jan_apr'
    elif (df['transactiondate'] >= pd.to_datetime('2017-05-01')) and (df['transactiondate'] < pd.to_datetime('2017-09-01')):
        return 'may_aug'
    else:
        return 'sept_dec'
    
# Writing a function to determine the sale time in Seasons.
def season(df):
    '''
    This function takes in a dataframe, reads a transactiondate column and returns a season group category.

    Parameters: df - a dataframe with a transactiondate column in the format "YEAR-MONTH-DAY"

    Returns: a season group category of either Winter, Spring, Summer, or Fall
    '''
    if (df['transactiondate'] >= pd.to_datetime('2017-01-01')) and (df['transactiondate'] < pd.to_datetime('2017-03-01')) \
        or (df['transactiondate']>= pd.to_datetime('2017-12-01')):
        return 'winter'
    elif (df['transactiondate'] >= pd.to_datetime('2017-03-01')) and (df['transactiondate'] < pd.to_datetime('2017-06-01')):
        return 'spring'
    elif (df['transactiondate'] >= pd.to_datetime('2017-06-01')) and (df['transactiondate'] < pd.to_datetime('2017-09-01')):
        return 'summer'
    else:
        return 'fall' 

def map_days(df):
    '''
    This function takes in a dataframe containing a transactiondate column and returns the dataframe with a new column called 'day_of_week'
    that gives the day of week of the transaction.

    Parameters: df - a dataframe with a transactiondate column in the format "YEAR-MONTH-DAY"

    Returns: a dataframe with a new column 'day_of_week' that shows which day the transactiondate took place on
    '''
    df['day_of_week'] = df.transactiondate.dt.day_of_week
    days = {0: 'monday',
                1: 'tuesday',
                2: 'wednesday',
                3: 'thursday',
                4: 'friday',
                5: 'saturday',
                6: 'sunday'}
    # map counties to fips codes
    df.day_of_week = df.day_of_week.map(days)
    return df

def encode_columns(df, columns_to_encode):
    '''
    This function takes in a  dataframe and using one-hot encoding, encodes categorical variables. It does not drop the original
    categorical columns. This is done purposefully to allow for easier Exploratory Data Analysis.  Removal of original categorical columns
    will be done in a separate function 'drop_pre_encoded' if needed..

    Parameters: df - a dataframe with the expected feature names and columns
    Returns: encoded - a dataframe with all desired categorical columns encoded.
    '''

    dummy_df = pd.get_dummies(df[columns_to_encode], drop_first=False)
    encoded = pd.concat([df, dummy_df], axis = 1)
    return encoded

def drop_pre_encoded(df, columns_to_drop):
    df.drop(columns = columns_to_drop, inplace=True)
    return df

def add_cluster_features(df):
    '''
    This function takes in a dataframe with specific columns needed and using KMeans clustering creates two synthetic
    columns, one based on season_summer and scaledyear_to_scaledtax, the other based on day_of_week_sunday and 
    season_fall.
    
    Parameters: df - a zillow dataframe that is ready to have features added.
    
    Returns: a dataframe with the added features
    '''
    
    X = df[['season_summer', 'scaledyear_to_scaledtax']]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    df['summer_scaledyear_to_scaledtax'] = kmeans.predict(X)
    
    X = df[['day_of_week_sunday', 'season_fall']]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    df['sunday_fall'] = kmeans.predict(X)
    
    return df

def add_new_columns(df):
    '''
    This function takes in a zillow dataframe and performs various column additions to it and then outputs the dataframe with the newly added columns.

    Parameters: df - a zillow dataframe that has been cleaned and ready for transformations/additions.

    Returns: a dataframe with new column additions added on the end
    '''
    temp_df = temp_data_scaler(df,['yearbuilt','taxamount'])
    df['scaledyear_to_scaledtax'] = temp_df.yearbuilt / temp_df.taxamount
    df = df[df.scaledyear_to_scaledtax != np.inf]
    df['home_price_per_sq_ft'] = df.structure_tax_value / df.square_footage
    df['land_price_per_lot_sq_ft'] = df.land_tax_value / df.lot_size

    df['four_month_range'] = df.apply(four_month_split,axis=1)
    df['season'] = df.apply(season,axis=1)
    df = map_days(df)
    df = encode_columns(df, ['four_month_range', 'season', 'day_of_week'])
    df = drop_pre_encoded(df, ['four_month_range', 'season', 'day_of_week'])
    
    df = add_cluster_features(df)
    return df
