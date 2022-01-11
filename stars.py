# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 15:08:27 2021

@author: admin
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df= pd.read_csv(r"C:\Users\admin\Desktop\machine_learning\desicion tree\Stars.csv")

a = ['Blue White','Blue white','Blue-White','white','Whitish','Yellowish White','yellowish','Yellowish','White-Yellow','Pale yellow orange','Orange-Red']
for i in range(len(df['Color'])):
    if df['Color'][i] in a[:3]:
        df['Color'][i] = 'Blue-white'
    elif df['Color'][i] in a[3:5]:
        df['Color'][i] = 'White'
    elif df['Color'][i] in a[5:9]:
        df['Color'][i] = 'yellow-white'
    elif df['Color'][i] in a[9:]:
        df['Color'][i] = 'Orange'

df1=pd.get_dummies(data=df,columns=["Color","Spectral_Class"],drop_first=True)

df1 = df1[['Temperature', 'L', 'R', 'A_M', 'Color_Blue-white',
       'Color_Orange', 'Color_Red', 'Color_White', 'Color_yellow-white',
       'Spectral_Class_B', 'Spectral_Class_F', 'Spectral_Class_G',
       'Spectral_Class_K', 'Spectral_Class_M', 'Spectral_Class_O', 'Type']]

scaler = StandardScaler()
scaler.fit(df1.drop('Type',axis = 1))
scaled_features = scaler.transform(df1.drop('Type',axis = 1))
df_feat = pd.DataFrame(scaled_features,columns = df1.columns[:-1])
X = df_feat
y = df1['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



print("Tree")
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)
tahmin=dtr.predict(X_test)
print(tahmin)
tree.plot_tree(dtr)
