import pandas as pd
#read csv
riceAMLObj = pd.read_csv('Rice_MSC_Dataset.csv', sep=',')

#features we are extracting from the dataset, excluding the image color features
featureList = ["AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ","SOLIDITY","CONVEX_AREA", "EXTENT","ASPECT_RATIO","ROUNDNESS","COMPACTNESS","SHAPEFACTOR_1","SHAPEFACTOR_2","SHAPEFACTOR_3","SHAPEFACTOR_4","CLASS"]

#drop irrelevant features 
for feature in riceAMLObj:
    if feature not in featureList:
        riceAMLObj = riceAMLObj.drop(labels=feature, axis=1)

#make the output classes binary
classList = ["Basmati", "Jasmine"]
riceAMLObj = riceAMLObj.loc[(riceAMLObj["CLASS"] == classList[0]) | (riceAMLObj["CLASS"] == classList[1])]
#print shape
print("Features and attributes of dataset: ", riceAMLObj.shape)

#train test split
y = riceAMLObj["CLASS"]
x = riceAMLObj.drop(labels="CLASS", axis=1)
testSize = 0.2

from sklearn.model_selection import train_test_split
import numpy as np
#create knn classifier object with 1 nearest neighbor and standard Euclidean metric
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize)

#logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, max_iter=100000)
lr.fit(x_train, y_train)

predictions = lr.predict(x_test)

print("Training score: ", lr.score(x_train, y_train))
print("Testing score: ", lr.score(x_test, y_test))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )
disp.plot()

plt.show()