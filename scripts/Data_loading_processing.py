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

cols_to_remove=['AGI', 'FEDTAX','PEARNVAL','PTOTVAL','TAXINC', ]


cols_to_drop=['MARSUPWT']



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

def drop_not_needed_cols(df, cols_to_drop)





train_data = load_data(train_data_path)
train_data=add_column_names_convert_target(train_data,column_names, cols_to_remove)

print(train_data.head(3))
