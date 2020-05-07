import numpy as np
import pandas as pd
import os,sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('parkinsons.data')
# print(df.head())
#data flair get the features and labels
features = df.loc[:,df.columns !='status'].values[:,1:]
labels = df.loc[:,'status'].values

print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#We have 147 ones and 48 zeros in the
# status column in our dataset.
scaler = MinMaxScaler((-1,1))
#scale=tawsi3
x=scaler.fit_transform(features)
y=labels
#split the dataset into training and testing
# sets keeping 20% of the data for testing.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)
# Initialize an XGBClassifier and train the model
model = XGBClassifier()
print("model is = ",x_test)
model.fit(x_train,y_train)
#Finally, generate y_pred (predicted values for x_test)
# and calculate the accuracy for the model


y_pred = model.predict(x_test)
print("y prediction = ",y_pred)
print(accuracy_score(y_test,y_pred)*100)









