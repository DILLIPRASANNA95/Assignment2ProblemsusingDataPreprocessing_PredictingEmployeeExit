import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
files.upload()

df=pd.read_csv('obesity.csv')
df.head()

df.drop(['YearEnd', 'LocationAbbr', 'Datasource',
       'Topic', 'Data_Value_Unit', 'Data_Value_Type',
        'Data_Value_Alt', 'Data_Value_Footnote_Symbol',
       'Data_Value_Footnote','Sample_Size', 'Total',
       'Race/Ethnicity', 'GeoLocation', 'ClassID', 'TopicID', 'QuestionID',
       'DataValueTypeID', 'LocationID', 'StratificationCategory1',
       'Stratification1', 'StratificationCategoryId1', 'StratificationID1'],1,inplace=True)

df.columns

df.drop_duplicates(inplace=True)

df.isna().sum()

for i in df.columns:
  if df[i].isna().sum()>36000:
    df.drop(i,1,inplace=True)

for i in df.columns:
  if df[i].isna().sum()>0:
    df=df[df[i].notna()]

df.isna().sum()

df.head()

df.YearStart.unique()

df.Question.unique()

df=df[df.Question=='Percent of adults aged 18 years and older who have obesity']

df.drop(['Class','Question'],1,inplace=True)

df.drop_duplicates(inplace=True)
len(df)

df.reset_index(drop=True, inplace=True)

df_grouped=df.groupby(['LocationDesc','YearStart']).mean().reset_index()
df_grouped

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

df_grouped.rename(columns = { 'YearStart':'Year',}, inplace = True)

df_grouped['location']=lb.fit_transform(df_grouped.LocationDesc)

x=df_grouped.drop('LocationDesc',1)
y=df_grouped.Data_Value
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)