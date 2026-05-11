# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:

Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2:

Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3:

Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4:

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5:

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6:

Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.



## PROGRAM

### Name: Dinesh V

### Register Number: 212224040076

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)


    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1306" height="270" alt="image" src="https://github.com/user-attachments/assets/8cb1d35a-d6d8-4042-8abe-176ba978e359" />


### OUTPUT
<img width="397" height="229" alt="image" src="https://github.com/user-attachments/assets/578d5d2c-3882-4877-adc6-efa330d8dd23" />

## Confusion Matrix
<img width="743" height="603" alt="image" src="https://github.com/user-attachments/assets/cc14b383-e6f9-4e96-8ecb-2e06fcb16b3e" />


## Classification Report
<img width="666" height="457" alt="image" src="https://github.com/user-attachments/assets/d0d7846d-6836-4c48-a68f-15f2864e188c" />


### New Sample Data Prediction
<img width="420" height="111" alt="image" src="https://github.com/user-attachments/assets/bdd11029-9046-464d-a403-fda14fcec1f6" />


## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
