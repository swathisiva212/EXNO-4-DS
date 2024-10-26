# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:  
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/86458501-01c9-49af-a25f-8cf8bfedcac9)

data.isnull().sum()

![image](https://github.com/user-attachments/assets/f9be3870-7194-4f4c-8b50-6008f61d668f)
````
missing=data[data.isnull().any(axis=1)]
missing
````
![image](https://github.com/user-attachments/assets/a6e7a698-049e-42a2-8d04-b3c015d24d5c)

````
data2=data.dropna(axis=0)
data2
````
![image](https://github.com/user-attachments/assets/80686430-6371-4319-8f01-b623d6697c8c)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/e17bd8f3-e140-4dde-b89a-79bab047cf0a)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/2cb788f0-a530-4961-962b-3f5325504986)
```

data2
````
![image](https://github.com/user-attachments/assets/703795f9-f7a1-441d-a3ef-1c353dd79246)
```

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/fd62a441-b7a4-491d-a6e5-feae44957a6a)

```
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/user-attachments/assets/c9e27cce-3362-45b1-b442-3a0b5dfa965a)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/801bc960-77bb-4175-9f87-9827d1ca98ba)


```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/62ac8d6e-94b3-41c6-812a-28c176b83af7)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/79d6cf44-edbd-4b75-b0d1-986cfa475db3)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
````
![image](https://github.com/user-attachments/assets/34091a90-27c0-423f-839b-beaca5048b32)
```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/c49d830a-b0a5-4df9-9a53-4330225a19dc)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/faf4347a-a8be-4701-9691-ed26e9fd6f3f)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/061dcd4b-2fc8-484d-b6dc-1f15639d5acd)
```
data.shape
![image](https://github.com/user-attachments/assets/cff3c6cc-539b-44f6-abfe-20a8a95dc50b)
```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
````
![image](https://github.com/user-attachments/assets/07f28155-0294-4dd9-b0a6-a4443f4edc98)

```

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
````
![image](https://github.com/user-attachments/assets/00e0920e-6627-488b-8fd6-d586f29f946a)
```

tips.time.unique()
```
![image](https://github.com/user-attachments/assets/01f6a57b-3790-4b0a-ab75-0510bd047f15)

```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
````
![image](https://github.com/user-attachments/assets/e0bc2671-8cd0-4196-b773-df8422e4e900)

# RESULT:
       # INCLUDE YOUR RESULT HERE
