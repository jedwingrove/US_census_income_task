import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

train_data_path='/Users/jed/Documents/JED/Dataiku/drive-download-20250106T224222Z-001/census_income_learn.csv'

column_names=['AAGE','ACLSWKR','ADTIND','ADTOCC','AGI','AHGA','AHRSPAY','AHSCOL','AMARITL',

            'AMJIND','AMJOCC','ARACE','AREORGN','ASEX','AUNMEM','AUNTYPE','AWKSTAT','CAPGAIN',
            'CAPLOSS','DIVVAL','FEDTAX','FILESTAT','GRINREG','GRINST','HHDFMX','HHDREL','MARSUPWT',
            'MIGMTR1','MIGMTR2','MIGMTR4','MIGSAME','MIGSUN','NOEMP','PARENT','PEARNVAL','PEFNTVTY',
            'PEMNTVTY','PENATVTY','PRCITSHP','PTOTVAL','SEOTR','TAXINC','VETQVA','VETYN','WKSWORK', 'YEAROFSUR']

cols_to_remove=['AGI', 'FEDTAX','PEARNVAL','PTOTVAL','TAXINC' ]

race_mapping_dict = {
    ' White': 'White',
    ' Black': 'non_White', 
    ' Asian or Pacific Islander': 'non_White',
    ' Other': 'non_White',
    ' Amer Indian Aleut or Eskimo': 'non_White'
}

education_mapping = {
' Less than 1st grade': 'Grade-school',
' 1st 2nd 3rd or 4th grade': 'Grade-school', 
' 5th or 6th grade': 'Grade-school', 
' 7th and 8th grade':'Grade-school', 
' 9th grade': 'HS-nongrad', 
' 10th grade': 'HS-nongrad',
' 11th grade': 'HS-nongrad', 
' 12th grade no diploma': 'HS-nongrad',
' High school graduate': 'HS-grad', 
' Associates degree-academic program': 'HS-grad',
' Associates degree-occup /vocational':'HS-grad',
' Some college but no degree': 'HS-grad',
' Bachelors degree(BA AB BS)': 'Graduate', 
' Masters degree(MA MS MEng MEd MSW MBA)': 'Graduate',
' Doctorate degree(PhD EdD)': 'Graduate',
' Prof school degree (MD DDS DVM LLB JD)': 'Graduate'
 }


cols_to_drop = ['AREORGN', ]

def load_data(file_path):
    return pd.read_csv(file_path, header=None)


def add_column_names_convert_target(df, col_names, cols_to_remove):
    for col in cols_to_remove:
        if col in col_names:
            col_names.remove(col)

    
    col_names=col_names+['TARGET']
    df.columns=col_names
    
    df['TARGET_bin']=np.where(train_data.TARGET==' 50000+.',1,0)
    df.drop('TARGET', axis=1, inplace=True)
    if len(column_names)== len(train_data.columns) == len(test_data.columns):
        print('Column names have been sucessffully added to the train datasets')

    return df

def drop_duplicates_and_conflicting_samples(df):
    df=df.drop('MARSUPWT', axis=1)
    df=df.drop_duplicates(keep='first')
    
    # Identify conflicting instances
    conflicting_instances = df[df.duplicated(subset=df.columns.difference(['TARGET_bin']), keep=False)]
    
    # drop all conflicting instances, not enough information to decide which one to keep
    df.drop_duplicates(subset=df.columns.difference(['TARGET_bin']), keep='first', inplace=True)
    
    return df

def drop_children_from_df(df, age_col_name, age_col_new_name):
    df=df[df[age_col_name] > 14] 
    df = df.copy()
    df[age_col_new_name] = df[age_col_name]
    # df.loc[:, age_col_new_name] = df[age_col_name]
    df.drop(age_col_name, axis=1, inplace=True)
    return df 

def binarise_sex(df, sex_col_name, sex_new_col_name):
    df[sex_new_col_name] = np.where(df[sex_col_name] == ' Male', 1, 0)
    df.drop(sex_col_name, axis=1, inplace=True)
    return df

def apply_race_mapping_and_drop_race_col(df, race_col_name, race_new_col_name, race_mapping):
    # apply new mapping of keywords using race_mapping dict
    df['race_mapped'] = df[race_col_name].replace(race_mapping)
    # make 1 where == White, and Zero for other 
    df[race_new_col_name] = np.where(df['race_mapped'] == 'White', 1, 0)

    # drop original column and temp col, as not needed 
    df.drop(race_col_name, axis=1, inplace=True)
    df.drop('race_mapped', axis=1, inplace=True)

    return df

def apply_education_mapping_and_drop_education_col(df, edu_col_name, edu_mapping):
    # apply new mapping of keywords using race_mapping dict
    df['edu_mapped'] = df[edu_col_name].replace(edu_mapping)
    tmp_var='edu_mapped'    
    
    #instantiate OneHotEncoder
    encoder = OneHotEncoder(sparse=False) 
    # fit and transform the column
    encoded = encoder.fit_transform(df[[tmp_var]])

    # get the column names
    column_names=[]
    for i in sorted(df[tmp_var].unique()):
        column_names.append('Education_{}'.format(i))

    encoded_df = pd.DataFrame(encoded, columns=column_names)

    # reset indexes for both df and encoded df  
    df = df.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    merged_df = df.join(encoded_df, how='left')

    # drop the tmp_var column
    df=merged_df.copy()
    df.drop(tmp_var, axis=1, inplace=True)
    df.drop(edu_col_name, axis=1, inplace=True)
    return df 



def parental_country_of_birth(df, father_col_name, mother_col_name, new_col_name):
    
    # first replace values so that it is ones for US and zeros for not
    df['US_father'] = np.where(df[father_col_name] == ' United-States', 1, 0)
    df['US_Mother'] = np.where(df[mother_col_name] == ' United-States', 1, 0)
    
    # add the two columns to create an ordinal series (0=neither US born, 1=1xparent, 2=both)
    df['Parents']=df['US_father'] + df['US_Mother']

    # replace values so that it is 1 for >=1 or 0 for 0
    df[new_col_name] = np.where(df['Parents'] >= 1, 1, 0)

    remove = [father_col_name, mother_col_name,'US_father','US_Mother','Parents','PENATVTY']
    df.drop(remove, axis=1, inplace=True)
    return df



 



def no_utility_cols_to_drop(df, cols_to_drop):
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

# def marking_cols_to_remove_not_in_universe(df):
#     for column in df.select_dtypes(include=['object']).columns:
#     # Calculate the percentage of 'Not in universe' for the current column
#     percentage = (df[column] == ' Not in universe').mean() * 100
#     percentage_dict[column] = percentage

#     # Sort the dictionary by percentages in descending order
#     sorted_percentage_dict = dict(sorted(percentage_dict.items(), key=lambda item: item[1], reverse=True))

#     # Print the columns and their percentages
#     for column, percentage in sorted_percentage_dict.items():
#         if percentage > 0:
#             print(f"{column}: {percentage:.2f}%")

#     # anything greater than %50 will be dropped
#     for column, percentage in sorted_percentage_dict.items():
#         if percentage > 50:
#             df.drop(column, axis=1, inplace=True)
#             print(column, 'dropped from analysis')





train_data = load_data(train_data_path)
train_data = add_column_names_convert_target(train_data,column_names, cols_to_remove)
train_data = drop_duplicates_and_conflicting_samples(train_data)
train_data = drop_children_from_df(train_data, 'AAGE', 'Age')
train_data = binarise_sex(train_data, 'ASEX', 'Male')
train_data = apply_race_mapping_and_drop_race_col(train_data, 'ARACE', 'Race_white', race_mapping_dict)
train_data = apply_education_mapping_and_drop_education_col(train_data, 'AHGA', education_mapping)
train_data = parental_country_of_birth(train_data, 'PEFNTVTY', 'PEMNTVTY', 'Parents_birth')


print(train_data.head(3))
print(len(train_data))
print(train_data.columns)
