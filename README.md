# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE

### DATA.CSV

```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```

### ENCODING.CSV

```

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4

```

TITANIC.CSV

```

import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5

```
# OUPUT:

### DATA CSV

# Initial Dataset:

![image](https://user-images.githubusercontent.com/94165327/195992349-fa887fb7-9f07-4229-9ef4-9c218728f4ba.png)


# Binary Encoding:

![image](https://user-images.githubusercontent.com/94165327/195992386-2e7d534b-8b37-4c7d-b4ec-e23292860e38.png)


![image](https://user-images.githubusercontent.com/94165327/195992403-62ca997d-2ba5-4beb-8d31-1a123c9eca43.png)


# Encoded Dataset:

![image](https://user-images.githubusercontent.com/94165327/195992439-cd53b4dd-82cd-474f-8ba1-30dab2a6b821.png)

# Data Scaling using MinMaxScaler:

![image](https://user-images.githubusercontent.com/94165327/195992476-c101be95-0b65-49a3-9017-817e3690a2bb.png)

# Data Scaling using StandardScaler:

![image](https://user-images.githubusercontent.com/94165327/195992514-bbf90b57-b1ac-410b-8961-3f7ea1f4ccdd.png)

# Data Scaling using MaxAbsScaler:

![image](https://user-images.githubusercontent.com/94165327/195992476-c101be95-0b65-49a3-9017-817e3690a2bb.png)

# Encoding.csv :

# Initial Dataset:

![image](https://user-images.githubusercontent.com/94165327/195992645-00577a48-63bc-4e5e-b12d-abcf1402025f.png)

#Binary Encoding:

![image](https://user-images.githubusercontent.com/94165327/195992674-cd8545f5-b250-41aa-a0ae-bb7c7881708c.png)

![image](https://user-images.githubusercontent.com/94165327/195992403-62ca997d-2ba5-4beb-8d31-1a123c9eca43.png)

# Encoded Dataset:

![image](https://user-images.githubusercontent.com/94165327/195992728-9f8ff164-20df-4201-b667-9e3b38b33456.png)

# Data Scaling using MinMaxScaler:

![image](https://user-images.githubusercontent.com/94165327/195992785-dcdc89ba-ad32-4345-a264-0c51f6cd51a0.png)

# Data Scaling using MaxAbsScaler:

![image](https://user-images.githubusercontent.com/94165327/195992881-7467f65e-d488-40e1-8d14-ee1cf3e52b74.png)

# Data Scaling using RobustScaler:

![image](https://user-images.githubusercontent.com/94165327/195992955-656eadf2-b160-4434-9145-34005de28c86.png)

# Titanic.csv :

# Initial Dataset:

![image](https://user-images.githubusercontent.com/94165327/195993035-83127fa8-3ece-4a20-aad1-fabd156cda89.png)

# Data cleaning before encoding:

![image](https://user-images.githubusercontent.com/94165327/195993066-d84abc51-aa75-4117-ab8e-af3e37085734.png)

![image](https://user-images.githubusercontent.com/94165327/195993066-d84abc51-aa75-4117-ab8e-af3e37085734.png)

![image](https://user-images.githubusercontent.com/94165327/195993083-06458550-9b11-450a-9075-c70c84d3e849.png)


# Cleaned Dataset:

![image](https://user-images.githubusercontent.com/94165327/195993131-c84bb69d-55b5-489d-a1a1-1216a13d7dc7.png)


# Binary Encoding:

![image](https://user-images.githubusercontent.com/94165327/195993155-e2b5cb5d-f495-4043-8788-18b4a861549b.png)


# Encoded Dataset:

![image](https://user-images.githubusercontent.com/94165327/195993244-eeff518b-3996-4867-b98c-25737c21992b.png)

# Data Scaling using MinMaxScaler:

![image](https://user-images.githubusercontent.com/94165327/195993286-c0c13073-1bcb-4551-a579-a8a886c61398.png)

# Data Scaling using StandardScaler:

![image](https://user-images.githubusercontent.com/94165327/195993339-ad92f302-a35a-4df1-94e0-6d12142c0173.png)


# Data Scaling using MaxAbsScaler:

![image](https://user-images.githubusercontent.com/94165327/195993361-75526508-9783-423d-b863-882611e9c4a6.png)


# Data Scaling using RobustScaler:

![image](https://user-images.githubusercontent.com/94165327/195993386-942fdf7d-ec43-481e-96c2-2cdc0131595e.png)


# RESULT:

Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

