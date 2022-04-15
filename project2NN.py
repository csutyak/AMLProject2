import torch
from torch import nn

class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linearLayer1 = nn.Linear(16, 32)
        self.linearLayer2 = nn.Linear(32, 32)
        self.linearLayer3 = nn.Linear(32, 5)
        self.dropoutLayer1 = nn.Dropout(0.1)
    def forward(self, input):
        hiddenLayer1 = nn.functional.relu(self.linearLayer1(input))
        hiddenLayer2 = nn.functional.relu(self.linearLayer2(hiddenLayer1))
        dropoutLayer1 = self.dropoutLayer1(hiddenLayer2 + hiddenLayer1)
        output = self.linearLayer3(dropoutLayer1)
        return output

model = ResidualNet()

#create optimizer 
from torch import optim
optimiser = optim.Adam(model.parameters(), lr=3e-4)

#loss function
lossFunction = nn.CrossEntropyLoss()

#loading data
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

from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np

#convert the different values of the labels to numbers
from enum import Enum
class riceType(Enum):
    Basmati = 0
    Arborio = 1
    Jasmine = 2
    Ipsala = 3
    Karacadag = 4
#convert to Np array
outputLabels = riceAMLObj['CLASS'].values
ctr = 0
for label in outputLabels:
    outputLabels[ctr] = riceType[label].value
    ctr += 1

outputLabels = torch.tensor(outputLabels.astype(np.int8)).type(torch.LongTensor)
inputFeatures = torch.tensor(riceAMLObj.drop(labels="CLASS", axis=1).values.astype(np.float32))
# Passing to DataLoader
train_data = TensorDataset(inputFeatures, outputLabels)

#train test split
train_ratio = 0.2
test_size = int(len(train_data)*train_ratio)
train_size = len(train_data)- int(len(train_data)*train_ratio)
train, test = random_split(train_data, [train_size, test_size])
train_loader = DataLoader(train, batch_size=32)
test_loader = DataLoader(test, batch_size=32)

#training loop
epochs = 30
for epoch in range(epochs):
    trainingLosses = list()
    testingLosses = list()
    ctr = 0
    for batch in train_loader:
        inputParams, outputLabel = batch

        #forward
        predictedLabel = model(inputParams)

        #compute objective function
        lossOutput = lossFunction(predictedLabel, outputLabel)

        # sets old gradients to 0
        model.zero_grad()

        #compute new gradients with partial derivatives
        lossOutput.backward()

        #back propogates
        optimiser.step()

        trainingLosses.append(lossOutput.item())

    for batch in test_loader:
        inputParams, outputLabel = batch

        #forward
        with torch.no_grad():
            predictedLabel = model(inputParams)

        #compute objective function
        lossOutput = lossFunction(predictedLabel, outputLabel)

        testingLosses.append(lossOutput.item())


    print("Epoch",epoch + 1," training loss: ", torch.tensor(trainingLosses).mean())
    print("Epoch",epoch + 1," testing loss: ", torch.tensor(testingLosses).mean())
    

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in test_loader:
    output = model(inputs) # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
    
    labels = labels.data.cpu().numpy()
    y_true.extend(labels) # Save Truth

# constant for classes
classes = ("Basmati" , "Arborio", "Jasmine" ,"Ipsala" , "Karacadag")

totalCorrect = 0
for index in range(len(y_pred)):
    if y_pred[index] == y_true[index]:
        totalCorrect += 1

accuracy = totalCorrect / len(y_pred)
print("Total Accuracy of model: ", accuracy)

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt='g', cmap="Greens")
plt.show()
