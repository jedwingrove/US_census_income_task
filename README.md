# US Census Personal Income Prediction 

## Dataset 
This project uses a Dataset extracted from the 1194/95 US Census. 

This data was extracted by a third party from the census bureau database found at https://www.census.gov/data.html

The data was split into train/test in approximately 2/3, 1/3 roportions using MineSet's MIndUtil mineset-to-mlc.

Basic statistics for this data set:
  Number of instances for training data = 199523
  Duplicate or conflicting instances : 46716
  Number of instances in test = 99762
  Duplicate or conflicting instances : 20936

Labels were provided for personal income greater than or less than $50,000:

Probability for the label '< $50000' : 93.80%
Probability for the label > $50000' : 6.20%

Number of attributes/feeatures = 40 (continuous : 7 nominal : 33)

In brief the data consists of:
Demographic Information (Age, Sex, Race, etc.)
Employment Information (Worker type, Occupation, Employment Status, etc.)
Income Information (Capital Gains, Dividends, etc.)
Geographical Information (Region, State, Migration Data)
Family and Household Information (Marital Status, Family Size, etc.)



## Objectives
To identify characteristics that are associated with a person making more or less than $50,000 per year income.



## Main Insights
Referring to the WIP.ipynb in notebooks: conducting EDA on the data, I have found a good number of features which look important in differentiating between high earners and low earners. 
Sligtly older, middle aged, male, white, well educated, employed. 

Use the Data_loading_processing.py script in scripts to complete data cleaning and feature engineering, changing the file path in file as needed. 

## Model Training 
I trained 3 models. Logistic regression, decision trees and random forest classifiers. 
Assessed using precision, recall, f1 score and AUC ROC. None of the models performed amazingly well but showed some improvement when setting class_imbalance='balanced'
## Model Selection 
This project will require further feature engineering and use of techniques to deal with class imbalance, such as SMOTE. 

## Model Exlpainability 
The logistic regression model had great interpretability showing features such as high levels of education, self employment and age as all being positively assoociated with higher perosnal income. Unemployment and lower educational attainment was negatively associated with higher earning. 

## Conclusions 
This was an interesting project, the data is limited in the explanation behind all the features. There is potential for further engineering of features and also parameter tuning for models and exploring new techniques. 
