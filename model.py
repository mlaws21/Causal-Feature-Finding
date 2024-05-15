import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Generate synthetic data
# np.random.seed(0)
# X = np.random.randn(100, 2)
# y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

def prepare_data(csvfile, num_train, outcome_name, n=None, subset=None):
    """
    Prepares the data in the specified CSV file for linear regression.
    
    Takes the name of a CSV file (e.g. ```small.easy.csv```) and the number 
    of desired training instances (which should be the first K training instances
    in the file)

    It should return 5 values:
      - a list of names of the features (in the order in which 
        they appear in the feature matrix -- the first name should be
        "offset")
      - the training feature matrix X_train
      - the training response vector y_train 
      - the test feature matrix X_test (everything not in train is in test)
      - the test response vector y_test

    """
    data = pd.read_csv(csvfile)
    feats_and_response = list(data.columns)
    
    feats = [x for x in feats_and_response if x != outcome_name]
    
    if subset is None:
        if n is not None and n < len(feats):
            feats = random.sample(feats, k=n)
    else:
        assert n == len(subset)
        feats = subset
        
    data.insert(1, "offset", [1.0] * len(data))
    feats.append("offset")
    response = outcome_name
    train_data = data[feats].head(num_train)
    xtrain = torch.tensor(train_data.values, dtype=torch.float32)
    ytrain = torch.tensor(data[response].head(num_train).values, dtype=torch.float32).unsqueeze(1)
    
    test_data = data[feats].tail(len(data) - num_train)
    xtest = torch.tensor(test_data.values, dtype=torch.float32)
    ytest = torch.tensor(data[response].tail(len(data) - num_train).values, dtype=torch.float32).unsqueeze(1)
    
    return feats, xtrain, ytrain, xtest, ytest
    

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class LinearRegression(nn.Module):
    def __init__(self, num_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)


def train(Xtrain, Ytrain, model, criterion=nn.BCELoss(), opt=optim.Adam, num_epochs=3000, verbose=True):
    
    # Training loop
    optimizer = opt(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(Xtrain)
        loss = criterion(outputs, Ytrain)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1000 == 0 and verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test the model

    return model


def plot(model, X, y):
    # plt.scatter(X, y)
    print(X)
    plt.scatter(y, model(X).detach().numpy(), color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.show()

def evaluate(model, Xtest, Ytest):
    with torch.no_grad():
        pred = model(Xtest)
        pred_class = (pred >= 0.5).float()
        accuracy = (pred_class == Ytest).float().mean()
        return accuracy.item()
        
    

def main():
    # feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("big_noise_binary.csv", 800, "Y", n=3, subset=["V6", "V7", "V3"])
    # feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("big_noise_binary.csv", 800, "Y")
    feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("big_noise_binary.csv", 800, "Y", n=3)
    
    
    
    # data = prepare_data("binary_small_noise.csv", 800, "Y", n=3, subset=None)
    # DiabetesPedigreeFunction
    # data = prepare_data("diabetes.csv", 600, "Outcome")
    # feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("binary_distribution_shift.csv", 800, "Y")
    
    
    print(feats)
    my_model = LogisticRegression(len(feats))
    trained_model = train(Xtrain, Ytrain, my_model)
    naive = torch.mean(Ytest).item()
    print("Naive:", max(naive, 1 - naive))
    eval = evaluate(trained_model, Xtest, Ytest)
    # plot(trained_model, data[-2], data[-1])
    print(f'Accuracy: {eval:.4f}')

if __name__ == "__main__":
    main()