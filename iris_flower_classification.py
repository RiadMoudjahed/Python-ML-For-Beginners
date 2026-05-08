import torch  # importing PyTorch library for deep learning
from torch._refs import zero  # importing zero function from PyTorch references
import torch.nn as nn  # importing neural network module from PyTorch
import torch.nn.functional as F  # importing functional interface for neural networks
import pandas as pd  # importing pandas library for data manipulation 
from sklearn.preprocessing import LabelEncoder  # importing label encoder for categorical data
from sklearn.model_selection import train_test_split  # importing function to split data into train/test sets
import matplotlib.pyplot as plt  # importing matplotlib for plotting
from sklearn.metrics import accuracy_score  # importing accuracy score metric

class Model(nn.Module):  # defining a neural network model class inheriting from nn.Module
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):  # constructor defining network architecture
        super().__init__()  # calling parent class constructor
        self.fc1 = nn.Linear(in_features, h1)  # first fully connected layer
        self.fc2 = nn.Linear(h1, h2)  # second fully connected layer
        self.out = nn.Linear(h2, out_features)  # output layer
    
    def forward(self, x):  # defining forward pass through network
        x = F.relu(self.fc1(x))  # applying ReLU activation after first layer
        x = F.relu(self.fc2(x))  # applying ReLU activation after second layer
        x = self.out(x)  # passing through output layer
        return x  # returning output

print ("\t"*2 + "Loading Data...")  # printing data loading message
df = pd.read_csv('iris.csv')  # loading iris dataset from CSV file
print ("\n")  # printing newline for spacing

X = df.drop(columns=["Id", "Species"])  # creating feature matrix by dropping ID and target columns
y = df["Species"]  # creating target vector with species labels
print (f"X shape: {X.shape}")  # displaying feature matrix dimensions
print (f"y shape: {y.shape}")  # displaying target vector dimensions
print ("\n")  # printing newline for spacing

encoder = LabelEncoder()  # initializing label encoder for categorical labels
y = encoder.fit_transform(y)  # encoding string labels to numerical values
print (f"y shape after encoding: {y.shape}")  # displaying encoded target shape
print ("\n")  # printing newline for spacing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # splitting data into training and testing sets
print (f"X_train shape: {X_train.shape}")  # displaying training features shape
print (f"X_test shape: {X_test.shape}")  # displaying testing features shape
print (f"y_train shape: {y_train.shape}")  # displaying training target shape
print (f"y_test shape: {y_test.shape}")  # displaying testing target shape
print ("\n")

X_train = torch.FloatTensor(X_train.values)  # converting training features to PyTorch float tensor
X_test = torch.FloatTensor(X_test.values)  # converting testing features to PyTorch float tensor
y_train = torch.LongTensor(y_train)  # converting training targets to PyTorch long tensor
y_test = torch.LongTensor(y_test)  # converting testing targets to PyTorch long tensor
print ("\n")  # printing newline for spacing

model = Model()  # creating instance of neural network model
print (model)  # displaying model architecture
print ("\n")

criterion = nn.CrossEntropyLoss()  # defining loss function for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # initializing Adam optimizer with learning rate 0.01

print ("\t" + "Training the model...")  # printing training start message
epochs = 100  # setting number of training epochs
losses = []  # initializing list to store loss values

for i in range(epochs):  # looping through each training epoch
    y_pred = model.forward(X_train)  # forward pass through network with training data
    loss = criterion(y_pred, y_train)  # calculating loss between predictions and actual labels
    losses.append(loss.item())  # storing loss value for plotting
    optimizer.zero_grad()  # clearing previous gradients
    loss.backward()  # computing gradients through backpropagation
    optimizer.step()  # updating model weights using gradients
    
print ("Losses:", losses)  # displaying all loss values from training
print ("\n")


plt.plot(range(epochs), losses)  # plotting loss values over epochs
plt.xlabel("Epochs")  # setting x-axis label
plt.ylabel("Loss")  # setting y-axis label
plt.title("Training Loss")  # setting plot title
plt.show()  # displaying the loss plot

with torch.no_grad():  # disabling gradient computation for evaluation
    y_pred = torch.argmax(model(X_test), dim=1)  # getting predicted class with highest probability
    print(y_pred)  # displaying predicted classes
print ("\n")

accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())  # calculating model accuracy
print (f"Accuracy: {accuracy}")  # displaying final accuracy score