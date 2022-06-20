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
    This function accesses a csv consisting of RSME results on train and validate subsets from models previously run and plots 
    them out; it highlights the mean results as a baseline and then shows the Polynomial Regressor model to be the best
    performing of them all.
    '''

    to_model = model_results[model_results.model != 'passive_aggressive_regressor'].drop(['cumulative_RMSE', 'diff_RMSE'], axis=1)

    plt.figure(figsize=(15,10))
    mod = sns.lineplot(data=to_model, markers=['.','.'],markersize=15)

    mod.axvline(model_results.model.iloc[0], label = 'Mean baseline', ls = ':', color = 'green')
    xymean = .15,.055
    mod.annotate(text = 'Mean baseline', xy = (xymean), size=13)

    mod.axvline(model_results.train_RMSE.idxmin(), label = 'Best Train Result', ls = ':', color = 'black')
    xytrain = model_results.train_RMSE.idxmin()+.15,model_results.train_RMSE.min()-.000025
    mod.annotate(text = model_results.loc[model_results.train_RMSE.idxmin()].model, xy = (xytrain), size=13)

    mod.axvline(model_results.validate_RMSE.idxmin(), label = 'Best Validate Result', ls = ':', color = 'purple')
    xyvalidate = model_results.validate_RMSE.idxmin()+.15,model_results.validate_RMSE.min()-.000025
    mod.annotate(text = model_results.loc[model_results.validate_RMSE.idxmin()].model, xy = (xyvalidate), size=13)

    plt.legend()
    mod.set(xticklabels=[])
    mod.set(xticks=[])
    plt.show()

def county_validate_plots(y_validate_la, y_validate_orange, y_validate_ventura):
    '''
    This function takes in a set of y_validate dataframes for Los Angeles, Orange and Ventura counties and produces a scatterplot of
    the resulting predictions vs actual data points for tax value target.
    '''
    plt.figure(figsize=(18, 7))
    plt.subplot(133)
    sns.scatterplot(data = y_validate_ventura, x = 'logerror', y = 'logerror_pred_lm2')
    sns.lineplot(x=(0,4000000), y=(0,4000000), color = '#FF5E13')
    plt.title('Ventura Validate subset', fontsize= '10')

    plt.subplot(131)
    sns.scatterplot(data = y_validate_la, x = 'logerror', y = 'logerror_pred_lm2')
    plt.ylabel([])
    sns.lineplot(x=(0,4000000), y=(0,4000000), color = '#FF5E13')
    plt.title('Los Angeles Validate subset', fontsize= '10')

    plt.subplot(132)
    sns.scatterplot(data = y_validate_orange, x = 'logerror', y = 'logerror_pred_lm2')
    sns.lineplot(x=(0,4000000), y=(0,4000000), color = '#FF5E13')
    plt.title('Orange Validate subset',fontsize= '10')

    plt.show()

def county_train_x_y(train_la, train_orange, train_ventura, features):
    '''
    This function takes in train dataframes split by county, as well as a list of features for the model, and splits the dataframes into X and y
    subsets for each county train dataframe.  It then outputs all of these split dataframes.

    Returns:    X_train_la - Dataframe with feature variables for modeling this specific county.
                y_train_la - Dataframes containing the target variable for modeling this specific county.
                X_train_orange - Dataframe with feature variables for modeling this specific county.
                y_train_orange - Dataframe containing target variable for modeling this specific county.
                X_train_ventura - Dataframe with feature variables for modeling this specific county.
                y_train_ventura - Dataframe containing target variable for modeling this specific county.
                features - a list of features to model on
    '''
    X_train_la = train_la[features]
    y_train_la = pd.DataFrame(train_la['logerror'])

    X_train_orange = train_orange[features]
    y_train_orange = pd.DataFrame(train_orange['logerror'])

    X_train_ventura = train_ventura[features]
    y_train_ventura = pd.DataFrame(train_ventura['logerror'])

    return X_train_la, y_train_la, X_train_orange, y_train_orange, X_train_ventura, y_train_ventura

def county_validate_x_y(validate_la, validate_orange, validate_ventura, features):
    '''
    This function takes in validate dataframes split by county, defines features for the model, and splits the dataframes into X and y
    subsets for each county validate dataframe.  It then outputs all of these split dataframes.

    Returns:    X_validate_la - Dataframe with feature variables for modeling this specific county.
                y_validate_la - Dataframes containing the target variable for modeling this specific county.
                X_validate_orange - Dataframe with feature variables for modeling this specific county.
                y_validate_orange - Dataframe containing target variable for modeling this specific county.
                X_validate_ventura - Dataframe with feature variables for modeling this specific county.
                y_validate_ventura - Dataframe containing target variable for modeling this specific county.
                features - a list of features to model on
    '''
    
    X_validate_la = validate_la[features]
    y_validate_la = pd.DataFrame(validate_la.logerror)

    X_validate_orange = validate_orange[features]
    y_validate_orange = pd.DataFrame(validate_orange.logerror)

    X_validate_ventura = validate_ventura[features]
    y_validate_ventura = pd.DataFrame(validate_ventura.logerror)

    return X_validate_la, y_validate_la, X_validate_orange, y_validate_orange, X_validate_ventura, y_validate_ventura


def county_test_x_y(test_la, test_orange, test_ventura, features):  
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
    X_test_la = test_la[features]
    y_test_la = pd.DataFrame(test_la.logerror)

    X_test_orange = test_orange[features]
    y_test_orange = pd.DataFrame(test_orange.logerror)

    X_test_ventura = test_ventura[features]
    y_test_ventura = pd.DataFrame(test_ventura.logerror)

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
    lm2.fit(X_train_degree2_la, y_train_la.logerror)

    # predict train
    y_train_la['logerror_pred_lm2'] = lm2.predict(X_train_degree2_la)

    # predict validate
    y_validate_la['logerror_pred_lm2'] = lm2.predict(X_validate_degree2_la)

    # predict test
    y_test_la['logerror_pred_lm2'] = lm2.predict(X_test_degree2_la)

    # evaluate: rmse on y_train_la
    rmse_train = mean_squared_error(y_train_la.logerror, y_train_la.logerror_pred_lm2)**(1/2)

    # evaluate: rmse on y_validate_la
    rmse_validate = mean_squared_error(y_validate_la.logerror, y_validate_la.logerror_pred_lm2)**(1/2)

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
    lm2.fit(X_train_degree2_orange, y_train_orange.logerror)

    # predict train
    y_train_orange['logerror_pred_lm2'] = lm2.predict(X_train_degree2_orange)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_orange.logerror, y_train_orange.logerror_pred_lm2)**(1/2)

    # predict validate
    y_validate_orange['logerror_pred_lm2'] = lm2.predict(X_validate_degree2_orange)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_orange.logerror, y_validate_orange.logerror_pred_lm2)**(1/2)

    # predict test
    y_test_orange['logerror_pred_lm2'] = lm2.predict(X_test_degree2_orange)

    print("\nRMSE for Polynomial Model for Orange county\n\nTraining/In-Sample: ", rmse_train, 
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
    lm2.fit(X_train_degree2_ventura, y_train_ventura.logerror)

    # predict train
    y_train_ventura['logerror_pred_lm2'] = lm2.predict(X_train_degree2_ventura)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_ventura.logerror, y_train_ventura.logerror_pred_lm2)**(1/2)

    # predict validate
    y_validate_ventura['logerror_pred_lm2'] = lm2.predict(X_validate_degree2_ventura)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_ventura.logerror, y_validate_ventura.logerror_pred_lm2)**(1/2)

    # predict test
    y_test_ventura['logerror_pred_lm2'] = lm2.predict(X_test_degree2_ventura)

    print("\nRMSE for Polynomial Model for Orange county\n\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)

    return y_train_ventura, y_validate_ventura, y_test_ventura


def county_models_test(y_test_la, y_test_orange, y_test_ventura):
    '''
    This function takes in y_test_la, y_test_orange, y_test_ventura dataframes which contain actual values and predictive values for the
    target variable based on the model created. It then prints out the RSME values for each model.
    '''

    y_test_orange = y_test_orange.drop(y_test_orange.logerror_pred_lm2.idxmax())
    rmse_test_ventura = mean_squared_error(y_test_ventura.logerror, y_test_ventura.logerror_pred_lm2)**(1/2)
    rmse_test_la = mean_squared_error(y_test_la.logerror, y_test_la.logerror_pred_lm2)**(1/2)
    rmse_test_orange = mean_squared_error(y_test_orange.logerror, y_test_orange.logerror_pred_lm2)**(1/2)

    print("\nRMSE for Polynomial Model for Los Angeles county\nTest/Out-of-Sample: ", rmse_test_la)
    print("\nRMSE for Polynomial Model for Orange county\nTest/Out-of-Sample: ", rmse_test_orange)
    print("\nRMSE for Polynomial Model for Ventura county\nTest/Out-of-Sample: ", rmse_test_ventura)



def validate_results_plot(model_results, y_validate_la, y_validate_orange, y_validate_ventura):
    '''
    This model plots the RSME results for the Mean Baseline, Aggregate Regressor Model, Los Angeles Model, Orange Model, Ventura Model
    '''
    aggregate_model_validation = model_results[model_results.model == 'polynomial_regression'].validate_RMSE.values[0]
    mean_model_validation = model_results[model_results.model == 'train_mean'].validate_RMSE.values[0]
    la_model_validation = mean_squared_error(y_validate_la.logerror, y_validate_la.logerror_pred_lm2)**(1/2)
    orange_model_validation = mean_squared_error(y_validate_orange.logerror, y_validate_orange.logerror_pred_lm2)**(1/2)
    ventura_model_validation = mean_squared_error(y_validate_ventura.logerror, y_validate_ventura.logerror_pred_lm2)**(1/2)

    plt.figure(figsize=(13,11))
    sns.barplot(x = ['Mean Baseline', 'Aggregate Regressor Model', 'Los Angeles Model', 'Orange Model', 'Ventura Model'], 
                y = [mean_model_validation, aggregate_model_validation, la_model_validation, orange_model_validation, ventura_model_validation], palette='flare')
    plt.axhline(mean_model_validation, color = '#38a4fc')
    plt.ylabel('Model RSME of logerror', size = 'large')
    plt.title('Error in Predicting logerror',size = 'x-large')
    plt.show()


def poly_regressor_model(X_train, X_validate, X_test, y_train, y_validate, validate_scaled, y_test):
        # making the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2, )

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model
    lm2 = LinearRegression(normalize=True)

    #fit the model
    lm2.fit(X_train_degree2, y_train.logerror)

    # predict train
    y_train['logerror_pred_lm2_ver2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm2_ver2)**(1/2)

    # predict validate
    y_validate['logerror_pred_lm2_ver2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm2_ver2)**(1/2)

    # predict test
    y_test['logerror_pred_lm2_ver2'] = lm2.predict(X_test_degree2)

    # This is just to predict the RMSE on validate for LA county
    la_only_validate_RMSE = mean_squared_error(y_validate[validate_scaled.county =='los_angeles'].logerror, y_validate[validate_scaled.county =='los_angeles'].logerror_pred_lm2_ver2)**(1/2)
    # This is just to predict the RMSE on validate for orange county
    orange_only_validate_RMSE = mean_squared_error(y_validate[validate_scaled.county =='orange'].logerror, y_validate[validate_scaled.county =='orange'].logerror_pred_lm2_ver2)**(1/2)
    # This is just to predict the RMSE on validate for ventura county
    ventura_only_validate_RMSE = mean_squared_error(y_validate[validate_scaled.county =='ventura'].logerror, y_validate[validate_scaled.county =='ventura'].logerror_pred_lm2_ver2)**(1/2)


    return la_only_validate_RMSE, orange_only_validate_RMSE, ventura_only_validate_RMSE, y_test

def comparison_plot(model_results, X_train, X_validate, X_test, y_train, y_validate, y_test, validate_scaled, y_validate_la, y_validate_orange, y_validate_ventura, ):
    la_only_validate_RMSE, orange_only_validate_RMSE, ventura_only_validate_RMSE, y_test = poly_regressor_model(X_train, X_validate, X_test, y_train, y_validate, validate_scaled, y_test)

    aggregate_model_validation = model_results[model_results.model == 'polynomial_regression'].validate_RMSE.values[0]
    mean_model_validation = model_results[model_results.model == 'train_mean'].validate_RMSE.values[0]
    la_model_validation = mean_squared_error(y_validate_la.logerror, y_validate_la.logerror_pred_lm2)**(1/2)
    orange_model_validation = mean_squared_error(y_validate_orange.logerror, y_validate_orange.logerror_pred_lm2)**(1/2)
    ventura_model_validation = mean_squared_error(y_validate_ventura.logerror, y_validate_ventura.logerror_pred_lm2)**(1/2)

    plt.figure(figsize=(13,11))
    g = sns.barplot(x = ['Mean Baseline', 'Aggregate Regressor Model', 'Los Angeles Model', 'Orange Model', 'Ventura Model'], 
                y = [mean_model_validation, aggregate_model_validation, la_model_validation, orange_model_validation, ventura_model_validation], palette='flare')

    g = sns.barplot(x = ['Mean Baseline', 'Aggregate Poly-Regressor Model', 'LA only', 'Orange only', 'Ventura only'], 
                y = [0, 0, la_only_validate_RMSE, orange_only_validate_RMSE, ventura_only_validate_RMSE], palette='viridis', alpha =.45)
    g.set(ylim=(.0425,.06))

    plt.title('Individual models RMSE vs Aggregate separated out',size = 'x-large')
    plt.axhline(mean_model_validation, color = '#38a4fc')
    plt.ylabel('Model RSME of logerror', size = 'large')

    plt.show()
    return y_test

def final_model_test(y_test, test_scaled):
    y_test = y_test.drop(y_test.logerror_pred_lm2_ver2.idxmax())
    rmse_aggregate_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_lm2_ver2)**(1/2)

    # This is just to predict the RMSE on validate for LA county
    la_only_test_RMSE = mean_squared_error(y_test[test_scaled.county =='los_angeles'].logerror, y_test[test_scaled.county =='los_angeles'].logerror_pred_lm2_ver2)**(1/2)
    # This is just to predict the RMSE on validate for orange county
    orange_only_test_RMSE = mean_squared_error(y_test[test_scaled.county =='orange'].logerror, y_test[test_scaled.county =='orange'].logerror_pred_lm2_ver2)**(1/2)
    # This is just to predict the RMSE on validate for ventura county
    ventura_only_test_RMSE = mean_squared_error(y_test[test_scaled.county =='ventura'].logerror, y_test[test_scaled.county =='ventura'].logerror_pred_lm2_ver2)**(1/2)

    print("\nRMSE for Aggregate Polynomial Model on \nTest/Out-of-Sample: ", rmse_aggregate_test)
    print("\n\nRMSE for Aggregate Polynomial Model for Los Angeles county on\nTest/Out-of-Sample: ", la_only_test_RMSE)
    print("\nRMSE for Aggregate Polynomial Model for Orange county on\nTest/Out-of-Sample: ", orange_only_test_RMSE)
    print("\nRMSE for Aggregate Polynomial Model for Ventura county on\nTest/Out-of-Sample: ", ventura_only_test_RMSE)

    return rmse_aggregate_test, la_only_test_RMSE, orange_only_test_RMSE, ventura_only_test_RMSE

def final_comparison_plot(model_results, X_train, X_validate, X_test, y_train, y_validate, y_test, validate_scaled, y_validate_la, y_validate_orange, y_validate_ventura, rmse_aggregate_test, la_only_test_RMSE, orange_only_test_RMSE, ventura_only_test_RMSE):
    # la_only_validate_RMSE, orange_only_validate_RMSE, ventura_only_validate_RMSE, y_test = poly_regressor_model(X_train, X_validate, X_test, y_train, y_validate, validate_scaled, y_test)
    y_test = y_test.drop(y_test.logerror_pred_lm2_ver2.idxmax())
    aggregate_model_validation = model_results[model_results.model == 'polynomial_regression'].validate_RMSE.values[0]
    mean_model_validation = model_results[model_results.model == 'train_mean'].validate_RMSE.values[0]
    la_model_validation = mean_squared_error(y_validate_la.logerror, y_validate_la.logerror_pred_lm2)**(1/2)
    orange_model_validation = mean_squared_error(y_validate_orange.logerror, y_validate_orange.logerror_pred_lm2)**(1/2)
    ventura_model_validation = mean_squared_error(y_validate_ventura.logerror, y_validate_ventura.logerror_pred_lm2)**(1/2)

    plt.figure(figsize=(13,11))
    g = sns.barplot(x = ['Mean Baseline', 'Aggregate Regressor Model', 'Los Angeles Model', 'Orange Model', 'Ventura Model'], 
                y = [mean_model_validation, aggregate_model_validation, la_model_validation, orange_model_validation, ventura_model_validation], palette='flare')

    g= sns.barplot(x = ['Mean Baseline', 'Aggregate Poly-Regressor Model', 'LA', 'Orange', 'Ventura'], 
                y = [0, rmse_aggregate_test, la_only_test_RMSE, orange_only_test_RMSE, ventura_only_test_RMSE], palette='viridis', alpha =.45)
    g.set(ylim=(.0425,.06))

    plt.title('Aggregate Model Validate vs Individual Models Validate vs Aggregate Model Test',size = 'x-large')
    plt.axhline(mean_model_validation, color = '#38a4fc')
    plt.ylabel('Model RSME of logerror', size = 'large')

    plt.show()
    return