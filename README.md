# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the necessary libraries to initiate the SVM python program.
2. Use chardet to dectect the characters if the csv file is encoded.
3. After all use pandas library to read the csv file.
4. Use the preprocess commands to know the dataset to found error if found do data preprocessing steps.
5. Select the features to give as input for x and y variable.
6. Use countvectorizer library from sklearn.feature_extraction.text to split the words in the sentence and its assigned the numerical value.
7. Use sklearn model selection library to split the datase for training and testing data.
8. Use sklearn.svm to get SVC for classification problems and the fit to train the model
9. Use y_pred to get the prediction of svc x_test and print the variable (y_pred).
10. Use sklearn.metrics to import accuracy_score to find the performance of thr model and print the acuarcy variable.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VESHWANTH.
RegisterNumber: 212224230300 
*/
import chardet
import pandas as pd
file="C:\\Users\\admin\\Downloads\\spam.csv"
with open(file,"rb") as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
data=pd.read_csv("C:\\Users\\admin\\Downloads\\spam.csv",encoding='Windows-1252')
data.head()
data.isnull().sum()
data.info()
data.shape
x.shape
y.shape
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape
y_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)
accuracy
```


## Output:
![1](https://github.com/user-attachments/assets/582e7ef5-0826-445a-be06-0fbbc1fd63b9)
![2](https://github.com/user-attachments/assets/d5157fca-f01e-4f37-8eb1-61f84a2c0184)
![3](https://github.com/user-attachments/assets/6dd40e47-563a-44c1-8c7c-19277cd44407)
![4](https://github.com/user-attachments/assets/a9e7194e-6639-4de2-8776-c300814346b2)
![5](https://github.com/user-attachments/assets/2be465d3-cea2-4e16-9bfa-65a8e6d94760)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
