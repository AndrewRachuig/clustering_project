import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def county_train_split(train_scaled):
    '''
    This function takes in a scaled train dataframe and returns 3 train dataframes split by county.
    '''
    # Separating train out by county

    train_la = train_scaled[train_scaled.county == 'los_angeles']
    train_orange = train_scaled[train_scaled.county == 'orange']
    train_ventura = train_scaled[train_scaled.county == 'ventura']

    return train_la, train_orange, train_ventura

def county_validate_split(validate_scaled):
    '''
    This function takes in a scaled validate dataframe and returns 3 validate dataframes split by county.
    '''
    # Separating validate out by county

    validate_la = validate_scaled[validate_scaled.county == 'los_angeles']
    validate_orange = validate_scaled[validate_scaled.county == 'orange']
    validate_ventura = validate_scaled[validate_scaled.county == 'ventura']

    return validate_la, validate_orange, validate_ventura

def county_test_split(test_scaled):
    '''
    This function takes in a scaled test dataframe and returns 3 test dataframes split by county.
    '''
    # Separating test out by county

    test_la = test_scaled[test_scaled.county == 'los_angeles']
    test_orange = test_scaled[test_scaled.county == 'orange']
    test_ventura = test_scaled[test_scaled.county == 'ventura']

    return test_la, test_orange, test_ventura


def plot_all_models(model_results):
    '''
    This function takes in a dataframe consisting of RSME results on train and validate subsets from models previously run and plots 
    them out; it highlights the mean and median results as a baseline and then shows the Polynomial Regressor model to be the best
    performing of them all.

    Arguments: A dataframe consisting of RSME results on train and validate subsets for various models.
    '''
    plt.figure(figsize=(10,10))
    mod = sns.scatterplot(data = model_results, palette='flare_r', s = 100, markers = ['o','o'])
    mod.set(xticklabels=[])
    mod.set(xticks=[])
    plt.text(5, 360000, "Polynomial Regression", horizontalalignment='center', size='medium', color='black', weight='normal')
    plt.text(0, 472000, "Mean", horizontalalignment='center', size='medium', color='black', weight='normal')
    plt.text(1, 496000, "Median", horizontalalignment='center', size='medium', color='black', weight='normal')
    plt.axhline(364500, color='black', lw =1, ls ='--')
    plt.show()

def county_validate_plots(y_validate_la, y_validate_orange, y_validate_ventura):
    '''
    This function takes in a set of y_validate dataframes for Los Angeles, Orange and Ventura counties and produces a scatterplot of
    the resulting predictions vs actual data points for tax value target.
    '''
    plt.figure(figsize=(18, 7))
    plt.subplot(133)
    sns.scatterplot(data = y_validate_ventura, x = 'taxvaluedollarcnt', y = 'taxvalue_pred_lm2')
    sns.lineplot(x=(0,4000000), y=(0,4000000), color = '#FF5E13')
    plt.title('Ventura Validate subset', fontsize= '10')

    plt.subplot(131)
    sns.scatterplot(data = y_validate_la, x = 'taxvaluedollarcnt', y = 'taxvalue_pred_lm2')
    plt.ylabel([])
    sns.lineplot(x=(0,4000000), y=(0,4000000), color = '#FF5E13')
    plt.title('Los Angeles Validate subset', fontsize= '10')

    plt.subplot(132)
    sns.scatterplot(data = y_validate_orange, x = 'taxvaluedollarcnt', y = 'taxvalue_pred_lm2')
    sns.lineplot(x=(0,4000000), y=(0,4000000), color = '#FF5E13')
    plt.title('Orange Validate subset',fontsize= '10')

    plt.show()

def county_train_x_y(train_la, train_orange, train_ventura):
    '''
    This function takes in train dataframes split by county, defines features for the model, and splits the dataframes into X and y
    subsets for each county train dataframe.  It then outputs all of these split dataframes.

    Returns:    X_train_la - Dataframe with feature variables for modeling this specific county.
                y_train_la - Dataframes containing the target variable for modeling this specific county.
                X_train_orange - Dataframe with feature variables for modeling this specific county.
                y_train_orange - Dataframe containing target variable for modeling this specific county.
                X_train_ventura - Dataframe with feature variables for modeling this specific county.
                y_train_ventura - Dataframe containing target variable for modeling this specific county.
    '''
    X_train_la = train_la[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_train_la = pd.DataFrame(train_la['taxvaluedollarcnt'])

    X_train_orange = train_orange[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_train_orange = pd.DataFrame(train_orange['taxvaluedollarcnt'])

    X_train_ventura = train_ventura[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_train_ventura = pd.DataFrame(train_ventura['taxvaluedollarcnt'])

    return X_train_la, y_train_la, X_train_orange, y_train_orange, X_train_ventura, y_train_ventura

def county_validate_x_y(validate_la, validate_orange, validate_ventura):
    '''
    This function takes in validate dataframes split by county, defines features for the model, and splits the dataframes into X and y
    subsets for each county validate dataframe.  It then outputs all of these split dataframes.

    Returns:    X_validate_la - Dataframe with feature variables for modeling this specific county.
                y_validate_la - Dataframes containing the target variable for modeling this specific county.
                X_validate_orange - Dataframe with feature variables for modeling this specific county.
                y_validate_orange - Dataframe containing target variable for modeling this specific county.
                X_validate_ventura - Dataframe with feature variables for modeling this specific county.
                y_validate_ventura - Dataframe containing target variable for modeling this specific county.
    '''
    
    X_validate_la = validate_la[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_validate_la = pd.DataFrame(validate_la.taxvaluedollarcnt)

    X_validate_orange = validate_orange[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_validate_orange = pd.DataFrame(validate_orange.taxvaluedollarcnt)

    X_validate_ventura = validate_ventura[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_validate_ventura = pd.DataFrame(validate_ventura.taxvaluedollarcnt)

    return X_validate_la, y_validate_la, X_validate_orange, y_validate_orange, X_validate_ventura, y_validate_ventura


def county_test_x_y(test_la, test_orange, test_ventura):
    '''
    This function takes in test dataframes split by county, defines features for the model, and splits the dataframes into X and y
    subsets for each county test dataframe.  It then outputs all of these split dataframes.

    Returns:    X_test_la - Dataframe with feature variables for modeling this specific county.
                y_test_la - Dataframes containing the target variable for modeling this specific county.
                X_test_orange - Dataframe with feature variables for modeling this specific county.
                y_test_orange - Dataframe containing target variable for modeling this specific county.
                X_test_ventura - Dataframe with feature variables for modeling this specific county.
                y_test_ventura - Dataframe containing target variable for modeling this specific county.
    '''
    X_test_la = test_la[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_test_la = pd.DataFrame(test_la.taxvaluedollarcnt)

    X_test_orange = test_orange[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_test_orange = pd.DataFrame(test_orange.taxvaluedollarcnt)

    X_test_ventura = test_ventura[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'yearbuilt']]
    y_test_ventura = pd.DataFrame(test_ventura.taxvaluedollarcnt)

    return X_test_la, y_test_la, X_test_orange, y_test_orange, X_test_ventura, y_test_ventura

def la_county_model(X_train_la, y_train_la, X_validate_la, y_validate_la, X_test_la, y_test_la):
    '''
    This function takes in all requisite dataframes for creating a polynomial model for Los Angeles county. It creates a Polynomial 
    Regression model, makes predictions and attaches those predictions to the y dataframes. It prints out the RSME values for train and
    validate subsets. Finally, it returns y_train_la, y_validate_la, y_test_la subsets to be used later if desired.
    '''
    # making the polynomial features to get a new set of features
    pf_la = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2_la = pf_la.fit_transform(X_train_la)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2_la = pf_la.transform(X_validate_la)
    X_test_degree2_la = pf_la.transform(X_test_la)

    # create the model
    lm2 = LinearRegression(normalize=True)

    # fit the model
    lm2.fit(X_train_degree2_la, y_train_la.taxvaluedollarcnt)

    # predict train
    y_train_la['taxvalue_pred_lm2'] = lm2.predict(X_train_degree2_la)

    # predict validate
    y_validate_la['taxvalue_pred_lm2'] = lm2.predict(X_validate_degree2_la)

    # predict test
    y_test_la['taxvalue_pred_lm2'] = lm2.predict(X_test_degree2_la)

    # evaluate: rmse on y_train_la
    rmse_train = mean_squared_error(y_train_la.taxvaluedollarcnt, y_train_la.taxvalue_pred_lm2)**(1/2)

    # evaluate: rmse on y_validate_la
    rmse_validate = mean_squared_error(y_validate_la.taxvaluedollarcnt, y_validate_la.taxvalue_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model for LA county\n\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)

    return y_train_la, y_validate_la, y_test_la

def orange_county_model(X_train_orange, y_train_orange, X_validate_orange, y_validate_orange, X_test_orange, y_test_orange):
    '''
    This function takes in all requisite dataframes for creating a polynomial model for Orange county. It creates a Polynomial 
    Regression model, makes predictions and attaches those predictions to the y dataframes. It prints out the RSME values for train and
    validate subsets. Finally, it returns y_train_la, y_validate_la, y_test_la subsets to be used later if desired.
    '''
    # making the polynomial features to get a new set of features
    pf_orange = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2_orange = pf_orange.fit_transform(X_train_orange)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2_orange = pf_orange.transform(X_validate_orange)
    X_test_degree2_orange = pf_orange.transform(X_test_orange)

    # create the model
    lm2 = LinearRegression(normalize=True)

    # fit the model
    lm2.fit(X_train_degree2_orange, y_train_orange.taxvaluedollarcnt)

    # predict train
    y_train_orange['taxvalue_pred_lm2'] = lm2.predict(X_train_degree2_orange)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_orange.taxvaluedollarcnt, y_train_orange.taxvalue_pred_lm2)**(1/2)

    # predict validate
    y_validate_orange['taxvalue_pred_lm2'] = lm2.predict(X_validate_degree2_orange)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_orange.taxvaluedollarcnt, y_validate_orange.taxvalue_pred_lm2)**(1/2)

    # predict test
    y_test_orange['taxvalue_pred_lm2'] = lm2.predict(X_test_degree2_orange)

    print("RMSE for Polynomial Model for Orange county\n\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)

    return y_train_orange, y_validate_orange, y_test_orange

def ventura_county_model(X_train_ventura, y_train_ventura, X_validate_ventura, y_validate_ventura, X_test_ventura, y_test_ventura):
    '''
    This function takes in all requisite dataframes for creating a polynomial model for Ventura county. It creates a Polynomial 
    Regression model, makes predictions and attaches those predictions to the y dataframes. It prints out the RSME values for train and
    validate subsets. Finally, it returns y_train_la, y_validate_la, y_test_la subsets to be used later if desired.
    '''
    # making the polynomial features to get a new set of features
    pf_ventura = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2_ventura = pf_ventura.fit_transform(X_train_ventura)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2_ventura = pf_ventura.transform(X_validate_ventura)
    X_test_degree2_ventura = pf_ventura.transform(X_test_ventura)

    # create the model
    lm2 = LinearRegression(normalize=True)

    # fit the model
    lm2.fit(X_train_degree2_ventura, y_train_ventura.taxvaluedollarcnt)

    # predict train
    y_train_ventura['taxvalue_pred_lm2'] = lm2.predict(X_train_degree2_ventura)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_ventura.taxvaluedollarcnt, y_train_ventura.taxvalue_pred_lm2)**(1/2)

    # predict validate
    y_validate_ventura['taxvalue_pred_lm2'] = lm2.predict(X_validate_degree2_ventura)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_ventura.taxvaluedollarcnt, y_validate_ventura.taxvalue_pred_lm2)**(1/2)

    # predict test
    y_test_ventura['taxvalue_pred_lm2'] = lm2.predict(X_test_degree2_ventura)

    print("RMSE for Polynomial Model for Orange county\n\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)

    return y_train_ventura, y_validate_ventura, y_test_ventura


def county_models_test(y_test_la, y_test_orange, y_test_ventura):
    '''
    This function takes in y_test_la, y_test_orange, y_test_ventura dataframes which contain actual values and predictive values for the
    target variable based on the model created. It then prints out the RSME values for each model.
    '''
    rmse_test_ventura = mean_squared_error(y_test_ventura.taxvaluedollarcnt, y_test_ventura.taxvalue_pred_lm2)**(1/2)
    rmse_test_la = mean_squared_error(y_test_la.taxvaluedollarcnt, y_test_la.taxvalue_pred_lm2)**(1/2)
    rmse_test_orange = mean_squared_error(y_test_orange.taxvaluedollarcnt, y_test_orange.taxvalue_pred_lm2)**(1/2)

    print("\nRMSE for Polynomial Model for Los Angeles county\nTest/Out-of-Sample: ", rmse_test_la)
    print("\nRMSE for Polynomial Model for Orange county\nTest/Out-of-Sample: ", rmse_test_orange)
    print("\nRMSE for Polynomial Model for Ventura county\nTest/Out-of-Sample: ", rmse_test_ventura)

def results_plot():
    '''
    This model plots the RSME results for the Mean Baseline, Aggregate Regressor Model, Los Angeles Model, Orange Model, Ventura Model
    '''
    plt.figure(figsize=(13,11))
    sns.barplot(x = ['Mean Baseline', 'Aggregate Regressor Model', 'Los Angeles Model', 'Orange Model', 'Ventura Model'], 
                y = [479022.1900, 365217.6465, 368333.99640893657, 309071.3777638292, 233689.5022497083], palette='flare')
    plt.axhline(365217, color = '#38a4fc')
    plt.ylabel('Model Error in $', size = 'large')
    plt.title('Error in Predicting Home Value',size = 'x-large')
    plt.show()