import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
files.upload()

df=pd.read_csv('train_set.csv')
len(df)

df.head()

df.drop_duplicates(inplace=True)

len(df)

df.isna().sum()

df=df[df.Union.notna()]
df=df[df.JF.notna()]

df.isna().sum()

df.dtypes

df.head()

for i in df.columns:
  print(i,':',set(list(df[i])))

group=df.groupby('OG').mean()
group

sns.scatterplot(x=df.Salaries,y=df.Total_Compensation)

group.columns

sns.scatterplot(x=df['H/D'],y=df.Total_Compensation)

sns.pointplot(y=group.index,x=group.Total_Compensation)

group2=df.groupby('YT').mean()
group2

sns.scatterplot(x=group2.index,y=group2.Total_Compensation,)

sns.scatterplot(x=df.Overtime,y=df.Total_Compensation)

df.corr()

newdf=df[['OGC','Salaries','H/D','Overtime','Total_Compensation']]
newdf.head()

dummies=pd.get_dummies(newdf.OGC,prefix='OGC')
dummies.head()

newdf=pd.concat([newdf,dummies],axis=1)
newdf.drop('OGC',axis=1,inplace=True)

newdf.head()

x=newdf.drop('Total_Compensation',axis=1)
y=newdf.Total_Compensation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
lreg=LinearRegression()

xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.2,random_state=0)

lreg.fit(xtrain,ytrain)
lreg.score(xtest,ytest)