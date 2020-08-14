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

train['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([train, test])

all_data.columns
```
## Step 1: Cleaning the data by removing/filling null values  

Checked for null values:
```
print('Train columns with null values:\n', train.isnull().sum())
print("---------------")

print('Test columns with null values:\n', test.isnull().sum())
print("---------------")

train.describe(include = "all")
```
Gave us 177 - Age, 687 - Cabin, 2 - Embarked null values in the train dataset, and  
86 - Age, 1 - Fare, 327 - Cabin null values in the test dataset  

Then I separated the numeric and categorical variables:
```
#Separate numeric and categorical variables

df_num = train[['Age', 'SibSp', 'Parch', 'Fare']]
df_cat = train[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]
```
And the distributions of the numeric variables:
```
#distributions for numeric variables 
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()
```
![Graph11](https://i.imgur.com/XIBawt4.png)  
![Graph12](https://i.imgur.com/ENmu0rX.png)  
![Graph13](https://i.imgur.com/XaSM5eN.png)  
![Graph14](https://i.imgur.com/2q3sQlI.png)  

Then I got the correlation heatmap to see how the different variables interacted:
```
#Correlation map to see how features are correlated with Survived and with each other
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
```
![Graph15](https://i.imgur.com/wwN65E4.png)  

This showed that the 'Age' and 'Fare' features are most correlated with the 'Pclass' feature, so grouping these features and then getting the median will get a more accurate result to replace the null values.

Next I did some box plots for the categorical variables:
```
for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()   
```
![Graph16](https://i.imgur.com/7tESTT9.png)  
![Graph17](https://i.imgur.com/5C22isQ.png)  
![Graph18](https://i.imgur.com/OPDBMfm.png)  
![Graph19](https://i.imgur.com/hJuqM6M.png)  
![Graph20](https://i.imgur.com/JfylLgL.png)  

## Feature Engineering: 
### The Cabin Catastrophe and Title Turmoil!  

The 'Cabin' variable had a lot of null values, and was very messy. I wanted to know if Cabin letter or how many people were in the cabin affected survival rate. So first I got the value counts for the number of cabins with different amount of people in them:

```
train['cabin_multiple'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

train['cabin_multiple'].value_counts()
```
>0 ------------ 687    
>1 ------------ 180    
>2 ------------ 16    
>3 ------------ 6    
>4 ------------ 2      
>Name: cabin_multiple, dtype: int64

Next I made a table to see if these numbers were related to survival rate at all:
```
pd.pivot_table(train, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket', aggfunc = 'count')
```
| Num | Died | Survived |  
| --- | ---- | -------- |
| 0 | 481.0 | 206.0 |
| 1 | 58.0 | 122.0 |
| 2 | 7.0 | 9.0 |
| 3 | 3.0 | 3.0 |
| 4 | NaN | 2.0 | 

Turns out having multiple people in a cabin actually helps with survival a lot!

Next to find out if the Cabin letters have anything to do with survival:
```
train['cabin_adv'] = train.Cabin.apply(lambda x: str(x)[0])
#n = null
print(train.cabin_adv.value_counts())
pd.pivot_table(train, index='Survived', columns='cabin_adv', values='Name', aggfunc='count')
```
>n ---- 687  
>C ---- 59  
>B ---- 47  
>D ---- 33  
>E ---- 32  
>A ---- 15  
>F ---- 13  
>G ---- 4  
>T ---- 1    
>Name: cabin_adv, dtype: int64  

| Letter | Died | Survived |
| ------ | ---- | -------- |
| A | 8.0 | 7.0 |
| B | 12.0 | 35.0 |
| C | 24.0 | 35.0 |
| D | 8.0 | 25.0 |
| E | 8.0 | 24.0 | 
| F | 5.0 | 8.0 |
| G | 2.0 | 2.0 |
| T | 1.0 | NaN |
| Null | 481.0 | 206.0 |

So simply having cabin data correlates to a higher survival rate.

Next I filled in the missing values for Age, Embarked, and Fare. For Age and Fare I grouped by Pclass (according to the heatmap it is the variable that most correlates) for a more accurate prediction:

```
#Filling in or removing missing values in train and test datasets

#Age and Pclass are most related, so will group by Pclass and then get the median age
all_data.Age = all_data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))
    
all_data.Embarked.fillna(all_data.Embarked.mode()[0], inplace = True)
    
#Fare and Pclass are most related, so once again groupby & get median
all_data.Fare = all_data.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

#delete the useless columns
drop_column = ['PassengerId', 'Ticket']
train.drop(drop_column, axis=1, inplace = True)

print(all_data.isnull().sum())
print("--------------")
print(test.isnull().sum())
```
## Step 2: Adding some new features:  

I created some new features.   
Firstly I created a 'FamilySize' feature by combining the Sibling Size and Parent Size features (+1).  
Secondly, I created the 'IsAlone' feature to show if the person was on the ship alone or not.  
Thirdly, I split the much less useful 'Name' feature to show just the person's title, as this is actually correlated to whether or not the person survived, unlike the 'name' feature. I also grouped the rarer titles into an 'Other' category. 

```
#Creating new features
#Getting FamilySize from Sibling and Parent counts
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

all_data['IsAlone'] = 1 
all_data['IsAlone'].loc[all_data['FamilySize'] > 1] = 0

all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

#Group rare title names into "Other"
stat_min = 10 
title_names = (all_data['Title'].value_counts() < stat_min) 

all_data['Title'] = all_data['Title'].apply(lambda x: 'Other' if title_names.loc[x] == True else x)
print(all_data['Title'].value_counts())
```
## Step 3: Fun Graph Time!

I created some cat plots of the new features (& some others) to see how they relate to survival:
```
sns.catplot(x="Sex", y="Survived", kind="bar", data=train)
sns.catplot(x="FamilySize", y="Survived", kind="bar", data=train)
sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=train)
``` 
![Graph21](https://i.imgur.com/uAuziLm.png)    
![Graph21](https://i.imgur.com/eUmD72F.png)    
![Graph21](https://i.imgur.com/d1U7MNw.png)  

Then I was done playin' around, so I unleashed the GRAPH KING:  
```
#ALL THE DATA!
pp = sns.pairplot(train, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
```
![Graph6](https://i.imgur.com/uUKdGiS.png)    

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

correlation_heatmap(train)
```
![Graph7](https://i.imgur.com/qFiTpSE.png)

## Step 4: Encoding  

Here I created the categorical variables from earlier, log transformed the 'Fare' feature to a more normal distribution, converted 'Pclass' to a category, and created the dummy variables of all the categories: 
```
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
       
# log transform of fare
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()


all_data.Pclass = all_data.Pclass.astype(str)

all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','FamilySize','IsAlone','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','Title','train_test']])
```
Then I split the data to run the models on and avoid overfitting:
```
#Splitting data to train test

X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)


y_train = all_data[all_data.train_test==1].Survived
y_train.shape
```
Then I scaled the dummies using StandardScaler(). Later on I tested the scaled versions against the normal versions and the scaled versions performed better!:
```
#Scaling data 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','norm_fare', 'FamilySize', 'IsAlone']]= scale.fit_transform(all_dummies_scaled[['Age','norm_fare','FamilySize', 'IsAlone']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived
```

## Step 5: Model Building!

The main purpose of working on this project was to learn model building and model tuning, as well as ensembling methods. I used Naive Bayes, Logistic Regression, Decision Tree, K Nearest Neighbour, Random Forest, Support Vector Classifier, and Xtreme Gradient Boosting:

```
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
```
I used the cross_val_score function to score each of these models.

**GaussianNB:**
```
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.7150838,  0.73595506, 0.75280899, 0.76404494, 0.80898876        
>0.7553763103383341

**Logistic Regression:**  
```
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.81564246, 0.8258427,  0.80898876, 0.8258427,  0.84831461   
>0.8249262444291006

**Logistic Regression Scaled:**
```
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.81564246, 0.8258427,  0.80898876, 0.8258427,  0.84831461  
>0.8249262444291006  

**Decision Tree:**
```
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.76536313, 0.76404494, 0.81460674, 0.75842697, 0.79775281   
>0.7800389178331555  

**Decision Tree Scaled:**
```
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.76536313, 0.76404494, 0.81460674, 0.75842697, 0.80337079   
>0.7811625133387735    

**K Nearest Neighbour:**
```
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.77094972, 0.7752809,  0.79775281, 0.81460674, 0.8258427
>0.7968865733475614  

**K Nearest Neighbour Scaled:**
```
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.79329609, 0.78651685, 0.81460674, 0.80898876, 0.85955056  
>0.8125918021467579   

**Random Forest:**
```
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.82122905, 0.79775281, 0.83146067, 0.74719101, 0.8258427    
>0.8046952482581131  

**Random Forest Scaled:**
```
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.82122905, 0.79775281, 0.83707865, 0.74719101, 0.8258427    
>0.805818843763731  

**Support Vector Classifier (Scaled):**
```
svc = SVC(probability = True)
cv = cross_val_score(svc,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.84357542, 0.8258427,  0.82022472, 0.80337079, 0.86516854    
>0.8316364321134895  

**Xtreme Gradient Boosting (Scaled):**
```
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.79888268, 0.80898876, 0.85393258, 0.78089888, 0.84269663    
>0.817079907099366  

**Voting Classifier:**
Now I used a voting classifier. This takes all the inputs and averages the result. For a 'hard' voting classifier, each model votes on whether it thinks the person survived or not and the result is just the most popular result. For a 'soft' voting classifier each model chooses it's percentage confidence, and the result is the averages of the confidence of each of the models. 

```
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft')     

cv = cross_val_score(voting_clf,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())
```
>0.82681564, 0.81460674, 0.8258427,  0.80337079, 0.87078652    
>0.8282844768062269  

## Step 6: Model Tuning!

Next I wanted to tune the models to get a more accurate prediction. I used GridSearchCV and RandomizedSearchCV for this:
```
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
```
First I defined a function that would report back the best score and best parameters of each model:
```
#Performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
```
Then it was time to plug in all the models and tune away!    

**Logistic Regression:**
```
lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train_scaled,y_train)
clf_performance(best_clf_lr,'Logistic Regression')
```
>Best Score: 0.8260498399347185  
>Best Parameters: {'C': 1.623776739188721, 'max_iter': 2000, 'penalty': 'l1', 'solver': 'liblinear'}  

**K Nearest Neighbour:**
```
knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train_scaled,y_train)
clf_performance(best_clf_knn,'KNN')
```
>Best Score: 0.8227167158370472    
>Best Parameters: {'algorithm': 'auto', 'n_neighbors': 9, 'p': 1, 'weights': 'uniform'}  

**Support Vector Classifier:**
```
svc = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train_scaled,y_train)
clf_performance(best_clf_svc,'SVC')
```
>Best Score: 0.8316364321134895    
>Best Parameters: {'C': 1, 'degree': 2, 'kernel': 'poly'}  

**Random Forest:**

For the Random Forest and XGBoost models, I used the RandomizedSearchCV, and then fine-tuned the models with GridSearchCV to save time:  
```
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [100,500,1000], 
                                  'bootstrap': [True,False],
                                  'max_depth': [3,5,10,20,50,75,100,None],
                                  'max_features': ['auto','sqrt'],
                                  'min_samples_leaf': [1,2,4,10],
                                  'min_samples_split': [2,5,10]}
                                  
clf_rf_rnd = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 100, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train_scaled,y_train)
clf_performance(best_clf_rf_rnd,'Random Forest')
```
>Best Score: 0.8338836231247255 
>Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 75, 'bootstrap': False}  

**Random Forest Fine Tuning:**
```
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [50,100,150,200,250],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [70, 75, 80, 90],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2, 3],
                                  'min_samples_split': [8, 9, 10, 11]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train_scaled,y_train)
clf_performance(best_clf_rf,'Random Forest')
```
>Best Score: 0.8406189190885694
>Best Parameters: {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 70, 'max_features': 10, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 100}

**Xtreme Gradient Boosting:**
```
xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [20, 50, 100, 250, 500,1000],
    'colsample_bytree': [0.2, 0.5, 0.7, 0.8, 1],
    'max_depth': [2, 5, 10, 15, 20, 25, None],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 1.5, 2],
    'subsample': [0.5,0.6,0.7, 0.8, 0.9],
    'learning_rate':[.01,0.1,0.2,0.3,0.5, 0.7, 0.9],
    'gamma':[0,.01,.1,1,10,100],
    'min_child_weight':[0,.01,0.1,1,10,100],
    'sampling_method': ['uniform', 'gradient_based']
}

clf_xgb_rnd = RandomizedSearchCV(xgb, param_distributions = param_grid, n_iter = 1000, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb_rnd = clf_xgb_rnd.fit(X_train_scaled,y_train)
clf_performance(best_clf_xgb_rnd,'XGB')
```
>Best Score: 0.8518485970748854  
>Best Parameters: {'subsample': 0.9, 'sampling_method': 'uniform', 'reg_lambda': 2, 'reg_alpha': 0.5, 'n_estimators': 500, 'min_child_weight': 0, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 1, 'colsample_bytree': 1}  

**Xtreme Gradient Boosting Fine Tuning:**
```
xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.85, 0.9, 0.95],
    'max_depth': [4, 5, 6, 7, 8],
    'reg_alpha': [0.5],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.85, 0.9, .95],
    'learning_rate':[0.5],
    'gamma':[0.5,1,2],
    'min_child_weight':[0],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train_scaled,y_train)
clf_performance(best_clf_xgb,'XGB')
```
>Best Score: 0.8563429790973573  
>Best Parameters: {'colsample_bytree': 0.85, 'gamma': 1, 'learning_rate': 0.5, 'max_depth': 5, 'min_child_weight': 0, 'n_estimators': 550, 'reg_alpha': 0.5, 'reg_lambda': 5, 'sampling_method': 'uniform', 'subsample': 0.9}  

Now I tried Ensembling. I experimented with many different types of estimators combos with hard and soft voting:

```
best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard') 
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft')
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft') 
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')

print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5).mean())

print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5).mean())

print('voting_clf_all :',cross_val_score(voting_clf_all,X_train,y_train,cv=5))
print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train,y_train,cv=5).mean())

print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5))
print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5).mean())
```
>voting_clf_hard : 0.82122905, 0.80898876, 0.83146067, 0.79213483, 0.85955056  
>voting_clf_hard mean : 0.8226727763480008  
>voting_clf_soft : 0.81005587, 0.80898876, 0.8258427,  0.82022472, 0.85393258  
>voting_clf_soft mean : 0.8238089259933463  
>voting_clf_all : 0.83240223, 0.80898876, 0.80898876, 0.8258427,  0.87078652  
>voting_clf_all mean : 0.8294017952419811  
>voting_clf_xgb : 0.83798883, 0.83146067, 0.84831461, 0.8258427,  0.86516854  
>voting_clf_xgb mean : 0.840631473228297  

Soft voting also lets you choose different weights for each of the models, so I also tested to see which weighting was best, and it turns out the original 1, 1, 1, waiting was already perfect.

```
params = {'weights' : [[1,1,1],[1,2,1],[1,1,2],[2,1,1],[2,2,1],[1,2,2],[2,1,2]]}

vote_weight = GridSearchCV(voting_clf_soft, param_grid = params, cv = 5, verbose = True, n_jobs = -1)
best_clf_weight = vote_weight.fit(X_train_scaled,y_train)
clf_performance(best_clf_weight,'VC Weights')
voting_clf_sub = best_clf_weight.best_estimator_.predict(X_test_scaled)
```
>Best Score: 0.8327600276191074  
>Best Parameters: {'weights': 1, 1, 1}

Next I made my predictions:
```
#Make Predictions 
voting_clf_hard.fit(X_train_scaled, y_train)
voting_clf_soft.fit(X_train_scaled, y_train)
voting_clf_all.fit(X_train_scaled, y_train)
voting_clf_xgb.fit(X_train_scaled, y_train)

best_rf.fit(X_train_scaled, y_train)
y_hat_vc_hard = voting_clf_hard.predict(X_test_scaled).astype(int)
y_hat_rf = best_rf.predict(X_test_scaled).astype(int)
y_hat_vc_soft =  voting_clf_soft.predict(X_test_scaled).astype(int)
y_hat_vc_all = voting_clf_all.predict(X_test_scaled).astype(int)
y_hat_vc_xgb = voting_clf_xgb.predict(X_test_scaled).astype(int)
```
After submission, the winner of 'best model' was surprising a tie!

The 'hard' voting classifier tied with 'Random Forest' with an accuracy of 78.468%, which scored in the top 20% of the leaderboard!

