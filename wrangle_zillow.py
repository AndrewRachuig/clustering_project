import pandas as pd
import numpy as np

import env
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_zillow(force_new=False):
    '''
    This function acquires the requisite zillow data from the Codeup SQL database and caches it locally it for future use in a csv 
    document; once the data is accessed the function then returns it as a dataframe.
    
    Arguments: force_new = Set to False by default; if set to true will force a pull from the SQL database and cache a local copy;
    this will overwrite any previously locally cached copy.
    
    Returns: A zillow dataframe ready to use.
    '''
    filename = "zillow.csv"
    query = '''
            SELECT 
                p.parcelid,
                airconditioningtypeid,
                architecturalstyletypeid,
                basementsqft,
                bathroomcnt,
                bedroomcnt,
                buildingclasstypeid,
                buildingqualitytypeid,
                calculatedbathnbr,
                decktypeid,
                finishedfloor1squarefeet,
                calculatedfinishedsquarefeet,
                finishedsquarefeet12,
                finishedsquarefeet13,
                finishedsquarefeet15,
                finishedsquarefeet50,
                finishedsquarefeet6,
                fips,
                fireplacecnt,
                fullbathcnt,
                garagecarcnt,
                garagetotalsqft,
                hashottuborspa,
                heatingorsystemtypeid,
                latitude,
                longitude,
                lotsizesquarefeet,
                poolcnt,
                poolsizesum,
                pooltypeid10,
                pooltypeid2,
                pooltypeid7,
                propertycountylandusecode,
                propertylandusetypeid,
                propertyzoningdesc,
                rawcensustractandblock,
                regionidcity,
                regionidcounty,
                regionidneighborhood,
                regionidzip,
                roomcnt,
                storytypeid,
                threequarterbathnbr,
                typeconstructiontypeid,
                unitcnt,
                yardbuildingsqft17,
                yardbuildingsqft26,
                yearbuilt,
                numberofstories,
                fireplaceflag,
                structuretaxvaluedollarcnt,
                taxvaluedollarcnt,
                assessmentyear,
                landtaxvaluedollarcnt,
                taxamount,
                taxdelinquencyflag,
                taxdelinquencyyear,
                censustractandblock,
                propertylandusedesc,
                logerror,
                p.transactiondate
            FROM
                (SELECT predictions_2017.parcelid, 
                        MAX(transactiondate) AS max_date
                FROM predictions_2017
                GROUP
                BY parcelid) AS m
            JOIN predictions_2017 as p
                ON p.parcelid = m.parcelid
                AND p.transactiondate = m.max_date
            JOIN
                properties_2017 ON properties_2017.parcelid = p.parcelid
            JOIN
                propertylandusetype USING (propertylandusetypeid)
            Where
                propertylandusedesc = 'Single Family Residential' AND 
                transactiondate LIKE '2017-%%'
            ;   
        '''
    url = env.get_db_url('zillow')
    
    if force_new == True:
        df = pd.read_sql(query, url)
        df.to_csv(filename, index = False)
    else:
        if os.path.isfile(filename):
            return pd.read_csv(filename)
        else:
            df = pd.read_sql(query, url)
            df.to_csv(filename, index = False)

    return df 

def obs_attr(df):
    num_rows_missing = []
    pct_rows_missing = []
    column_name = []
    for column in df.columns.tolist():
        num_rows_missing.append(df[column].isna().sum())
        pct_rows_missing.append(df[column].isna().sum() / len(df))
        column_name.append(column)
    new_info = {'column_name':column_name, 'num_rows_missing': num_rows_missing, 'pct_rows_missing': pct_rows_missing}
    return pd.DataFrame(new_info, index=None)

def drop_undesired(df, prop_required_column = .95, prop_required_row = .95):
    ''' This function takes in a dataframe and drops columns based on whether it meets the threshold for having values
    in the column and not null values. It then drops any rows based on whether it meets the threshold for having enough
    values in the row.
    
    Arguments: df - a dataframe
                prop_required_column - the proportion of a given column that must be filled by values and not nulls
                prop_required_row - the proportion of a given row that must be filled by values and not nulls
    Returns: a dataframe which no longer has the rows and columns dropped that didn't meet the threshhold.
    '''
    for column in df.columns.tolist():
        if 1-(df[column].isna().sum() / len(df)) < prop_required_column:
            df = df.drop(column, axis = 1)
            
    for row in range(len(df)):
        if 1-(df.loc[row].isna().sum() / len(df.loc[row])) < prop_required_row:
            df = df.drop(row, axis=0)
    return df


def minimum_sqr_ft(df):
    '''
    Function that takes in a dataframe and finds the minimum sq footage necessary given an input number of bathrooms and bedrooms.
    
    Arguments: A dataframe containing bathroomcnt and bedroomcnt columns.

    Returns: a total minimum amount of square feet necessary for a specified house.
    '''
    # min square footage for type of room
    bathroom_min = 10
    bedroom_min = 70
    
    # total MIN sqr feet
    total = (df.bathroomcnt * bathroom_min) + (df.bedroomcnt * bedroom_min)
    # return MIN sqr feet
    return total

def clean_sqr_feet(df):
    '''
    Takes in a dataframe finds the theoretical minimum sq footage given bathroom and bedroom inputs and compares that to the actual
    given sq footage.  
    Returns a dataframe where containing results only having an actual sq footage larger than the calculate minimum.
    '''
    # get MIN sqr ft
    min_sqr_ft = minimum_sqr_ft(df)
    # return df with sqr_ft >= min_sqr_ft
    # change 'sqr_ft' to whichever name you have for sqr_ft in df
    return df[df.calculatedfinishedsquarefeet >= min_sqr_ft]

def map_counties(df):
    # identified counties for fips codes 
    counties = {6037: 'los_angeles',
                6059: 'orange',
                6111: 'ventura'}
    # map counties to fips codes
    df.fips = df.fips.map(counties)
    df.rename(columns=({ 'fips': 'county'}), inplace=True)
    return df

def not_outlier(df_column, thresh=3.5):
    """
    Returns a boolean array with True if points are not outliers and False 
    otherwise.

    Parameters:
    -----------
        df_column : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(df_column.shape) == 1:
        df_column = np.array(df_column).reshape(-1,1)
    median = np.median(df_column, axis=0)
    diff = np.sum((df_column - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score < thresh

def drop_remaining(df):
    """
    This function takes in a zillow dataframe and drops unwanted columns / rows due to nulls or invalid data. It returns a dataframe with these columns removed.
    """
    df.drop(columns = ['rawcensustractandblock','rawcensustractandblock','roomcnt','finishedsquarefeet12','regionidcounty','regionidcity','censustractandblock','parcelid',
        'propertylandusedesc','assessmentyear', 'propertylandusetypeid','propertycountylandusecode','calculatedbathnbr','fullbathcnt','regionidzip'], inplace=True)
    df = df[df.lotsizesquarefeet.notnull()]
    df = df[df.yearbuilt.notnull() & df.taxamount.notnull() & df.structuretaxvaluedollarcnt.notnull()]

    return df

def convert_dtypes(df):
    '''
    This function takes in a zillow dataframe and changes the datatype for specific columsn to save on memory. It returns the modified dataframe.
    '''
    df.bedroomcnt = df.bedroomcnt.astype('uint8')
    df.bathroomcnt = df.bathroomcnt.astype('float16')
    df.yearbuilt = df.yearbuilt.astype('uint16')
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype('uint16')
    df['transactiondate'] = pd.to_datetime(df.transactiondate)
    return df

def clean_zillow(df):
    '''
    This function takes in an uncleaned zillow dataframe and peforms various cleaning functions. It returns a cleaned zillow dataframe.
    '''
    # Dropping undesired columns due to nulls
    df = drop_undesired(df)

    # Getting rid of unwanted columns
    df = drop_remaining(df)

    # Getting rid of invalid, wrong, or incorrectly entered data
    df = df[df.bedroomcnt != 0]

    # Getting rid of nonsense entries where the house has a sq footage value smaller than a theoretical minimum
    df = clean_sqr_feet(df)

    # Changing the fips column in the dataframe to show actual counties represented by fips number
    df = map_counties(df)

    # Changing datatypes for selected columns to improve efficiency
    df = convert_dtypes(df)

    # Running the not_outlier function to get rid of outliers with a modified z score of greater than a set threshhold. 
    # Keeps the vast majority of data and makes it more applicable.
    df = df[not_outlier(df.taxvaluedollarcnt, thresh=4.5)]
    df = df[not_outlier(df.calculatedfinishedsquarefeet, thresh=5)]
    df = df[not_outlier(df.lotsizesquarefeet, thresh=4)]
    df = df[not_outlier(df.bathroomcnt, thresh=4)]
    df = df[not_outlier(df.bedroomcnt, thresh=4)]
    df = df[not_outlier(df.logerror, thresh = 4)]

    df.rename(columns={"bathroomcnt": "bathrooms", "bedroomcnt": "bedrooms", "taxvaluedollarcnt": "tax_value", "calculatedfinishedsquarefeet": "square_footage",
        "lotsizesquarefeet": "lot_size", "structuretaxvaluedollarcnt": "structure_tax_value", "landtaxvaluedollarcnt": "land_tax_value"}, inplace=True)

    return df

def split_zillow(df):
    '''
    Takes in a cleaned zillow dataframe, splits it into train, validate and test subgroups and then returns those subgroups.
    
    Arguments: df - a cleaned pandas dataframe with the expected feature names and columns in the zillow dataset
    Return: train, validate, test - dataframes ready for the exploration and model phases.
    '''
    train, test = train_test_split(df, train_size = 0.8, random_state = 1234)
    train, validate = train_test_split(train, train_size = 0.7, random_state = 1234)
    return train, validate, test

def data_scaler(train, validate, test, columns_to_scale):
    '''
    This function takes in train, validate, test subsets of the cleaned zillow dataset and using the train subset creates a min_max 
    scaler. It thens scales the subsets and returns the train, validate, test subsets as scaled versions of the initial data.

    Arguments:  train, validate, test - split subsets from of the cleaned zillow dataframe
                columns_to_scale - a list of column names to scale
    Return: scaled_train, scaled_validate, scaled_test - dataframe with scaled versions of the initial unscaled dataframes 
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])

    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    return train_scaled, validate_scaled, test_scaled

def prepare_all_data(df, columns_to_scale):
    '''
    This function takes in a cleaned and prepared zillow dataframe. It then performs a train, validate, test split on the dataframe.
    Following it also makes scaled copies of the train, validate, test splits. Finally it returns the train subset to be used in EDA, 
    as well as validate and test even though these will probably not be used since modeling should be done on scaled data; additionally 
    it returns the train_scaled, validate_scaled, test_scaled subsets for modeling use.

    Arguments:  df - a cleaned, and prepared zillow dataframe
                columns_to_scale - a list of columns whose data should be scaled for modeling

    Returns:    train - a subset of the zillow dataframe to be used for Exploratory Data Analysis
                validate - a subset of the zillow dataframe (probably will not be used)
                test -  a subset of the zillow dataframe (probably will not be used)
                train_scaled - a subset of the zillow dataframe to be used for training an ML model
                validate_scaled - a subset of the zillow dataframe to be used for validation of an ML model 
                test_scaled - a subset of the zillow dataframe to be used for final testing of final ML model
    '''
    train, validate, test = split_zillow(df)
    train_scaled, validate_scaled, test_scaled = data_scaler(train, validate, test, columns_to_scale)

    return train, validate, test, train_scaled, validate_scaled, test_scaled