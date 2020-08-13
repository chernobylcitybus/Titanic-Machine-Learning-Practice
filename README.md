# Titanic-Machine-Learning-Practice
Trying to get more familiar with machine learning using the Titanic Dataset (Predicting likelihood of death)

First things first, reading the datasets and importing:
```
import numpy as np 
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt  
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn import model_selection, ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore the annoying warnings (from sklearn and seaborn)
color = sns.color_palette()
sns.set_style('darkgrid')

train = pd.read_csv('C:/Users/Luke/Downloads/titanic/train.csv')
test = pd.read_csv('C:/Users/Luke/Downloads/titanic/test.csv')

train_copy = train.copy(deep = True)

data_cleaner = [train_copy, test]
```
## Step 1: Cleaning the data by removing/filling null values  

Checked for null values:
```
print('Train columns with null values:\n', train_copy.isnull().sum())
print("---------------")

print('Test columns with null values:\n', test.isnull().sum())
print("---------------")

train.describe(include = "all")
```
Gave us 177 - Age, 687 - Cabin, 2 - Embarked null values in the train dataset, and  
86 - Age, 1 - Fare, 327 - Cabin null values in the test dataset  

To fill these null values I could have filled them with their respective median/mode values, but I wanted a more accurate result, so first I graphed a correlation heatmap:
```
#Correlation map to see how features are correlated with Survived and with each other
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
```
![Graph1](https://i.imgur.com/70yQTkP.png)

This showed that the 'Age' and 'Fare' features are most correlated with the 'Pclass' feature, so grouping these features and then getting the median will get a more accurate result. The 'Cabin' feature has too much data missing for an accurate result, so I dropped it.

```
#Filling in or removing missing values in train and test datasets
#Age and Pclass are most related, so will fill in Age with the median wrt. Pclass

for dataset in data_cleaner:
    #Age and Pclass are most related, so will group by Pclass and then get the median age
    dataset['Age'] = dataset.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
   
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    
    #Fare and Pclass are most related, so once again groupby & get median
    dataset['Fare'] = dataset.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    
#delete the useless columns
drop_column = ['PassengerId','Cabin', 'Ticket']
train_copy.drop(drop_column, axis=1, inplace = True)
```
## Step 2: Adding some new features:  

I created 5 new features.   
Firstly I created a 'FamilySize' feature by combining the Sibling Size and Parent Size features (+1).  
Secondly, I created the 'IsAlone' feature to show if the person was on the ship alone or not.  
Thirdly, I split the much less useful 'Name' feature to show just the person's title, as this is actually correlated to whether or not the person survived, unlike the 'name' feature. I also grouped the rarer titles into an 'Other' category.
Fourth and Fifthly, I created 'Fare' and 'Age' bins to make some later analysis a bit easier.  

```
#Creating new features
for dataset in data_cleaner:    
    #Getting FamilySize from Sibling and Parent counts
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    
#Group rare title names into "Other"
stat_min = 10 
title_names = (train_copy['Title'].value_counts() < stat_min) 

train_copy['Title'] = train_copy['Title'].apply(lambda x: 'Other' if title_names.loc[x] == True else x)
```
## Step 3: Encoding  

Here I used LabelEncoder() to convert the objects to categories for train and test datasets.  
I also defined the target variable ('Survived') that I needed to predict, the feature variables that I wanted to select from, and the x variable with bin features.  
```
#Converting data with Label Encoder

label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

#Defining the variable we want to predict
Target = ['Survived']

#Defining all the X (feature) variables that we want to select from
train_copy_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] 
train_copy_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] 
train_copy_xy =  Target + train_copy_x
print('Original X Y: ', train_copy_xy, '\n')

#define x variables with bin features 
train_copy_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
train_copy_xy_bin = Target + train_copy_x_bin
print('Bin X Y: ', train_copy_xy_bin, '\n')
```
>Original X Y:  ['Survived', 'Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']    
>Bin X Y:  ['Survived', 'Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']  

## Step 4: Splitting  

Next I used sklearn's train_test_split() function to split the test data to avoid overfitting:
```
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train_copy[train_copy_x_calc], train_copy[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(train_copy[train_copy_x_bin], train_copy[Target] , random_state = 0)
```
## Step 5: Fun Graph Time!

Next up it was time to get a better look at the data and see how all the variables fit together!  
First up, I made bar, point, and cat plots to see how some of the features correlated to our target variable ('Survived')  

```
#graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=train_copy, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train_copy, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=train_copy, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=train_copy, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=train_copy, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=train_copy, ax = saxis[1,2])

sns.catplot(x="Sex", y="Survived", kind="bar", data=train_copy)
sns.catplot(x="FamilySize", y="Survived", kind="bar", data=train_copy)

sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=train_copy)
```
![Graph2](https://i.imgur.com/VJBctt3.png)  
![Graph3](https://i.imgur.com/6ZqPGjg.png)  
![Graph4](https://i.imgur.com/dBlHZQJ.png)  
![Graph5](https://i.imgur.com/a1aq1nE.png)  

I was done playin' around, so I unleashed the GRAPH KING:
```
#ALL THE DATA!
pp = sns.pairplot(train_copy, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
```
![Graph6](https://i.imgur.com/u0CJX5n.png)  

Finally, I created another heatmap with my new features, including the percentage correlation numbers:
```
#correlation heatmap 
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Correlation Heatmap', y=1.05, size=15)

correlation_heatmap(train_copy)
```
![Graph7](https://i.imgur.com/ceJFews.png)

## Step 6: Machine Learning:

I decided to use the Random Forest model, mainly because I hadn't used it yet and wanted to figure out how it worked. Later on I tried LinearSVC, as recommended by sklearn's handy dandy flowchart:
![Graph8](https://scikit-learn.org/stable/_static/ml_map.png)

```
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=6, random_state=0)
random_forest.fit(train_copy[train_copy_x_bin], train_copy[Target])
Y_pred = random_forest.predict(test[train_copy_x_bin])
test['Survived'] = Y_pred
random_forest.score(train_copy[train_copy_x_bin], train_copy[Target])
acc_random_forest = round(random_forest.score(train_copy[train_copy_x_bin], train_copy[Target]) * 100, 2)
acc_random_forest
```
>85.19

RandomForest gave an accuracy score of 85.19% -- not bad!

LinearSVC:
```
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

clf = make_pipeline(StandardScaler(), LinearSVC())
clf.fit(train_copy[train_copy_x_bin], train_copy[Target])
prediction = clf.predict(test[train_copy_x_bin])

test['Survived'] = prediction
clf.score(train_copy[train_copy_x_bin], train_copy[Target])
acc_clf = round(clf.score(train_copy[train_copy_x_bin], train_copy[Target]) * 100, 2)
acc_clf
```
>79.57
LinearSVC gave an accuracy score of 79.57% -- quite a bit worse than RandomForest









