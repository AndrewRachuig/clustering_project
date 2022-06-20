# <center>Using Clustering to Predict Logerror of the Zestimate</center>

<img src="img/zillow_cluster.jpg" width=800 height=600 />

## Project Summary

### Project Objectives

- Provide a Jupyter Notebook with the following:
    - Data Acquisition
    - Data Preparation
    - Exploratory Data Analysis
        - Statistical testing where needed for exploration of data
        - Creating of new features through clutering modeling (KMeans) with appropriate statistical tests
    - Model creation and refinement
    - Model Evaluation
    - Conclusions and Next Steps
- Creation of python modules to assist in the entire process and make it easily repeatable
    - wrangle.py
    - explore.py (if necessary)
    - model.py (if necessary)
- Ask exploratory questions of the data that will give an understanding about the attributes and drivers of the zestimate logerror for Single Family Properties in 2017    
    - Answer questions through charts and statistical tests
- Construct the best possible model for predicting logerror of the zestimate
    - Make predictions for a subset of out-of-sample data
- Adequately document and annotate all code
- Give a 5 minute presentation to a Data Science Team
- Field questions about my specific code, approach to the project, findings and model

### Business Goals
- Construct an ML Regression model that predicts logerror of Single Family Properties using features from the Zillow Dataset. If possible construct features manually through intuition of the data and also by finding groups through ML clustering (KMeans).
- Uncover what the drivers of the logerror in the zestimate value of for single family properties.
- Deliver a report to the data science team that will consist of a notebook demo of the discoveries I made.
- Make recommendations on what works or doesn't work in logerror of the zestimate.

### Audience
- Target audience for my final report is a data science team.

### Project Deliverables
- A github readme explaining the project
- A jupyter notebook containing my final report (properly documented and commented)
- All modules containing functions created to achieve the final report notebook (wrangle.py, explore.py, model.py files)
- Other supplemental artifacts created while working on the project (e.g. wrangle/exploratoration/modeling notebook(s), images for the final report, etc.)
- Live presentation of final report notebook
---
### Data Dictionary

Target|Datatype|Definition|
|:-------|:--------|:----------|
| logerror | 42136 non-null: float64 | Zillow's log-error between their Zestimate and the actual sale price |

The following are features I used in my final model.

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| bathrooms       | 42136 non-null: float16 |     Number of bathrooms in home including fractional bathrooms |
| bedrooms        | 42136 non-null: uint8 |     Number of bedrooms in home  |
| square_footage  | 42136 non-null: uint16 |    Calculated total finished living area of the home  |
| county          | 42136 non-null: object |    The county in which the residence is located; from fips data  |
| latitude        | 42136 non-null: float64 | The latitude coordinates of the property|
| longitude       | 42136 non-null: float64 | The longitude coordinates of the property|
| lot_size        | 42136 non-null: float64 |     Area of the lot in square feet |
| yearbuilt       | 42136 non-null: uint16 |     The Year the principal residence was built  | 
| structure_tax_value    | 42136 non-null: float64  |     The assessed value of the built structure on the parcel  |
| tax_value       | 42136 non-null: float64  |     The total tax assessed value of the parcel  |
| land_tax_value  | 42136 non-null: float64  |     The assessed value of the land area of the parcel  |
| taxamount       | 42136 non-null: float64  |     The total property tax assessed for that assessment year  |
| transactiondate | 42136 non-null: datetime64  |     The date the principal residence was sold  |
| scaledyear_to_scaledtax | 42136 non-null: float64  |     The ratio of a scaled yearbuilt value to a scaled taxamount value  |
| home_price_per_sq_ft | 42136 non-null: float64  |     The structure_tax_value of the home divided by the residence square_footage area   |
| land_price_per_lot_sq_ft | 42136 non-null: float64  |     The land_tax_value of the property divided by the lot_size area  |
| four_month_range_jan_apr | 42136 non-null: uint8  |     The date the residence was sold categorized as sold between Jan-Apr or not  |
| four_month_range_may_aug | 42136 non-null: uint8  |     The date the residence was sold categorized as sold between May-Aug or not  |
| four_month_range_sept_dec | 42136 non-null: uint8  |     The date the residence was sold categorized as sold between Sept-Dec or not  |
| season_fall     | 42136 non-null: uint8  |     The date the residence was sold categorized between in the Fall or not  |
| season_spring   | 42136 non-null: uint8  |     The date the residence was sold categorized between in the Spring or not  |
| season_summer   | 42136 non-null: uint8  |     The date the residence was sold categorized between in the Summer or not  |
| season_winter   | 42136 non-null: uint8  |     The date the residence was sold categorized between in the Wiinter or not  |
| day_of_week_friday | 42136 non-null: uint8  |     The principal residence was sold on Friday or not  |
| day_of_week_monday | 42136 non-null: uint8  |     The principal residence was sold on Monday or not  |
| day_of_week_saturday | 42136 non-null: uint8  |     The principal residence was sold on Saturday or not  |
| day_of_week_sunday | 42136 non-null: uint8  |     The principal residence was sold on Sunday or not  |
| day_of_week_thursday | 42136 non-null: uint8  |     The principal residence was sold on Thursday or not  |
| day_of_week_tuesday | 42136 non-null: uint8  |     The principal residence was sold on Tuesday or not  |
| day_of_week_wednesday | 42136 non-null: uint8  |     The principal residence was sold on Wednesday or not  |
| summer_scaledyear_to_scaledtax   | 42136 non-null: int32  |     Synthesized feature from clustering: Based on features above  |
| sunday_fall     | 42136 non-null: int32  |     Synthesized feature from clustering: Based on features above: Sold on Sunday in the Fall or not  |

---
### Questions/thoughts I have of the Data
- What features are most strongly correlated to logerror of the zestimate?
- My guess is that whatever helped predict tax value of home in the previous project will similarly be predictive here.
- I think tax value of home will be the biggest predictor of the logerror.
- I'm going to feature engineer some columns.

---
## Project Plan and Data Science Pipeline

#### Plan
- **Acquire** data from the Codeup SQL Database. 
    - Initial inquiry into the data to see the shape and layout of things.
- Clean and **prepare** data for the explore phase. Create wrangle.py to store functions I create to automate the full process of acquisition, cleaning and preparation. Separate train, validate, test subsets and scaled data.
- Begin **exploration** of the data and ask questions leading to clarity of what is happening in the data. 
    - Find interactions between independent variables and the target variable using visualization and statistical testing.
    - Use clustering to explore the data. Provide a conclusion, supported by statistical testing and visualization on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.
- At least 4 different **models** are created and their performance is compared.
    - Evaluate models on train and validate datasets. Do further hyperparamter tuning to find the best performing models.
    - Choose the model with that performs the best. Do any final tweaking of the model. Automate modeling functions and put them into a model.py file.
- Evaluate final model on the test dataset.
- Construct Final Report notebook wherein I show how I arrived at the final regression model by using my created modules. Throughout the notebook, document conclusions, takeaways, and next steps.
- Create README.md with data dictionary, project and business goals, initial hypothesis and an executive summary
---
#### Plan &rarr; Acquire
- Create wrangle.py to store all functions needed to acquire dataset (and later prepare/clean dataset).
- Investigate the data in the Codeup SQL Database to determine what data to pull before actually acquiring the data.
- Retrieve data from the Database by running an SQL query to pull requisite Zillow data, and put it into a usable Pandas dataframe.
- Do cursory data exploration/summarization to get a general feel for the data contained in the dataset.
- Use the wrangle.py file to import and do initial exploration/summarization of the data in the Final Report notebook
---
#### Plan &rarr; Acquire &rarr; Prepare
- Explore the data further to see where/how the data is dirty and needs to be cleaned. This is not EDA. This is exploring individual variables so as to prepare the data to undergo EDA in the next step
- Use wrangle.py to store all functions needed to clean and prepare the dataset
    - A function which cleans the data:
        - Convert datatypes where necessary: objects to numerical; numerical to objects
        - Deal with missing values and nulls
        - Drop superfluous, erroneous or redundant data
        - Handle redundant categorical variables that can be simplified
        - Change names to snake case where needed
        - Drop duplicates
    - A function which splits the dataframe into 3 subsets: Train, Validate, and Test to be used for Exploration of the data next
    - A function which creates a scaled version of the 3 subsets: Train, Validate, and Test to be used for modeling later
- Use the wrangle.py file to import and do initial cleaning/preparation of the data in the Final Report notebook
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore
- Do Exploratory Data Analysis using bivariate and multivariate stats and visualizations to find interactions in the data
- Explore my key questions and discover answers to my hypotheses by running statistical analysis on data
- Find key features to use in the model. Similarly find unnecessary features which can be dropped
    - Look for correlations, relationships, and interactions between various features and the target
    - Understanding how features relate to one another will be key to understanding if certain features can or should be dropped/combined
- Find interactions between independent variables and the target variable using visualization and statistical testing.
- Use clustering to explore the data. Provide a conclusion, supported by statistical testing and visualization on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.
- Document all takeaways and answers to questions/hypotheses
- Create an explore.py file which will store functions made to aid in the data exploration
- Use explore.py and stats testing in the final report notebook to show how I came to the conclusions about which data to use
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore &rarr; Model
- Do any final pre-modeling data prep (drop/combine columns) as determined most beneficial from the end of the Explore phase
- Find and establish baseline RSME base on Mean and Median values of the train subset. This will give me an RSME level to beat with my models
- Create at least four separate models to predict logerror of the zestimate.
    - Given time attempt other models.
- For all models made, compare RSME results from train to validate
    - Look for hyperparamters that will give better results.
- Compare results from models to determine which is best. Chose the final model to go forward with
- Put all necessary functions for modeling functions into a model.py file
- Use model.py in the final report notebook to show how I reached the final model
- Having determined the best possible model, test it on out-of-sample data (the scaled test subset created in prepare) to determine accuracy
- Summarize and visualize results. Document results and takeaways
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore &rarr; Model &rarr; Deliver
- After introduction, briefly summarize (give an executive summary of) the project and goals before delving into the final report notebook for a walkthrough.
- Do not talk through my entire process in the initial pipeline. This is only the "good stuff" to show how I arrived at the model I did.
- Detail my thoughts as I was going through the process; explain the reasons for my choices.
---
## Executive Summary
- The features found to be key drivers of the logerror for Single Family Properties were:
    - Certain seasons in which the house was sold was a good predictor: Spring and Fall.
    - Yearbuilt to tax amount ratio
    - Certain days of the week in which the house was sold was a good predictor: namely Sunday.
- Again, homes in Los Angeles county were more disparate than homes on either Orange or Ventura counties and thus every model produced worse predictions for homes in Los Angeles county than Ventura or Orange counties.

**Recommendations**

- The further exploration of the data and subsequent modeling show that while certain groups were helpful in estimating logerror more accurately, the differences were small and not really that predictive. 
    - I would need further time to continue exploring to find any meaningful clusters; though it should be noted that after iterating through about 50000 cluster combinations, it's possible there are not any signficiant clusters which would help increase predictive power of a model for logerror.
- Further examination of the Aggregate Model vs Individual Model for results is needed.
---
## Reproduce this project
- In order to run through this project yourself you will need your own env.py file that wrangle.py will import to be able to access the Codeup database. You must have your own credentials to access the Codeup database
- All other requisite files for running the final project notebook (python files, images, csv files) are contained in this repository.
