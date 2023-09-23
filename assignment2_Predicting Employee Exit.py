import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('https://raw.githubusercontent.com/tranghth-lux/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')
df.head()

left=df[df.left==1]
len(left)

retained=df[df.left==0]
len(retained)

sns.histplot(df,x='salary',hue='left',multiple="dodge",stat='count',shrink=0.8)

sns.histplot(df,y='sales',hue='left',multiple="dodge",stat='count',shrink=0.8)

df.groupby('left').mean()

newdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
newdf.head()

dummies.head()

newdf=pd.concat([newdf,dummies],axis=1)
newdf.drop('salary',axis=1,inplace=True)
newdf.head()

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

cross_val_score(LogisticRegression(),x,y).mean()

cross_val_score(SVC(),x,y).mean()

cross_val_score(DecisionTreeClassifier(),x,y).mean()