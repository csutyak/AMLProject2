import pandas as pd
#read csv
riceAMLObj = pd.read_csv('Rice_MSC_Dataset.csv', sep=',')

#features we are extracting from the dataset, excluding the image color features
featureList = ["AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ","SOLIDITY","CONVEX_AREA", "EXTENT","ASPECT_RATIO","ROUNDNESS","COMPACTNESS","SHAPEFACTOR_1","SHAPEFACTOR_2","SHAPEFACTOR_3","SHAPEFACTOR_4","CLASS"]

#drop irrelevant features 
for feature in riceAMLObj:
    if feature not in featureList:
        riceAMLObj = riceAMLObj.drop(labels=feature, axis=1)

print("Features and attributes of dataset: ", riceAMLObj.shape)

#train test split
y = riceAMLObj["CLASS"]
x = riceAMLObj.drop(labels="CLASS", axis=1)
testSize = 0.2

from sklearn.model_selection import train_test_split
import numpy as np
#create knn classifier object with 1 nearest neighbor and standard Euclidean metric
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize)

#scale the dataset using mean and std; do training/testing separately to avoid data leakage problem
standardize = True
if standardize:
    x_train = (x_train - x_train.mean())/x_train.std()
    x_test = (x_test - x.mean())/x.std()

from sklearn import svm
clfsvm = svm.SVC(kernel='rbf')
clfsvm.fit(x_train, y_train)
print("kernal: " , 'rbf')
print("train Score: " , clfsvm.score(x_train, y_train))
print("test Score: " , clfsvm.score(x_test, y_test))
print("-----")


