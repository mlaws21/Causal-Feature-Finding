from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier


# Generate synthetic data
# np.random.seed(0)
# X = np.random.randn(100, 2)
# y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

def prepare_data(csvfile, num_train, outcome_name, offset=True, n=None, subset=None):
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
        feats = list(subset)
    
    if offset:
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
    
def eval_logistic(model, Xtest, Ytest):
    with torch.no_grad():
        model.eval()
        pred = model(Xtest)
        pred_class = (pred >= 0.5).float()
        accuracy = (pred_class == Ytest).float().mean()
        return round(accuracy.item(), 3)

def eval_tree(model, Xtest, Ytest):
    count = 0
    predictions = model(Xtest)
    ground = Ytest.flatten()

    assert len(predictions) == len(ground)
    for i in range(len(predictions)):
        if predictions[i] == ground[i]:
            count += 1

    return round(count / len(predictions), 3)


class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
    def fit(self, Xtrain, Ytrain, criterion=nn.BCELoss(), opt=optim.Adam, num_epochs=3000, verbose=True):
    
    # Training loop
    
        optimizer = opt(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(Xtrain)
            loss = criterion(outputs, Ytrain)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 1000 == 0 and verbose:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def evaluate(self, Xtest, Ytest):
        return eval_logistic(self, Xtest, Ytest)


class TreeTemplate():
    def __init__(self, classifier):
        self.tree = classifier
        
    def fit(self, Xtrain, Ytrain, verbose=False):
        self.tree.fit(Xtrain, Ytrain.flatten())
        
    def __call__(self, Xtest):
        return self.tree.predict(Xtest)
    
    def evaluate(self, Xtest, Ytest):
        return eval_tree(self, Xtest, Ytest)
    
class DecisionTree(TreeTemplate):
    def __init__(self, max_depth=3, criterion='gini'):
        super().__init__(
            tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion))
    
class BoostedDecisionTree(TreeTemplate):
    def __init__(self,  n_estimators=100, max_depth=3, criterion='friedman_mse'):
        super().__init__(
            GradientBoostingClassifier(max_depth=max_depth, criterion=criterion, n_estimators=n_estimators))

class BaggedDecisionTree(TreeTemplate):
    def __init__(self,  n_estimators=100, max_depth=None, criterion='gini'):
        super().__init__(
             BaggingClassifier(tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion), n_estimators=n_estimators))

class RandomForrest(TreeTemplate):
    def __init__(self,  n_estimators=100, max_depth=None, criterion='gini'):
        super().__init__(
             RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion))

def nn_helper(num_inputs, num_hidden, hidden_size, activation=nn.ReLU()):
    """
    A two-layer feedforward neural network with two input features,
    H hidden features, and two response values.
    
    init_bound should be set for all dense layers as specified
    by the init_bound argument (which will be a float). 

    activation is a torch.Module that performs the activation
    function (e.g. ReLU(), LeakyReLU(a), TaLU).
    
    """

    model = nn.Sequential()
    model.add_module(f"dense1", nn.Linear(in_features=num_inputs, out_features=hidden_size))
    model.add_module(f"drop1", nn.Dropout(p=0.5))
    model.add_module(f"activation1", activation)
    
    for i in range(num_hidden - 1):
        model.add_module(f"dense{i+2}", nn.Linear(in_features=hidden_size, out_features=hidden_size))
        model.add_module(f"drop{i+2}", nn.Dropout(p=0.5))
        model.add_module(f"activation{i+2}", activation)
        

    model.add_module(f"dense{num_hidden+1}", nn.Linear(in_features=hidden_size, out_features=1))
    model.add_module("sigmoid", nn.Sigmoid())

    return model

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden=3, hidden_size=64, activation=nn.ReLU()):
        super(NeuralNetwork, self).__init__()
        self.model = nn_helper(num_inputs, num_hidden, hidden_size, activation=activation)
    
    def forward(self, x):
        return self.model.forward(x)
    def fit(self, Xtrain, Ytrain, criterion=nn.BCELoss(), opt=optim.Adam, num_epochs=1000, verbose=True):
        self.train()
    # Training loop
        optimizer = opt(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(Xtrain)
            loss = criterion(outputs, Ytrain)
            
            # Backward pass and optimization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0 and verbose:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    def evaluate(self, Xtest, Ytest):
        return eval_logistic(self, Xtest, Ytest)


# clf = tree.DecisionTreeClassifier(max_depth=None)
# clf = GradientBoostingClassifier(n_estimators=100)
# clf = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=None), n_estimators=100)
# clf = RandomForestClassifier(n_estimators=100)


# clf.fit(Xtrain, Ytrain.flatten())

# # tree.plot_tree(clf)
# count = 0
# predictions = clf.predict(Xtest)
# ground = Ytest.flatten()

# assert len(predictions) == len(ground)
# for i in range(len(predictions)):
#     if predictions[i] == ground[i]:
#         count += 1


# #         count += 1
# print(count / len(predictions))



def plot(model, X, y):
    # plt.scatter(X, y)
    print(X)
    plt.scatter(y, model(X).detach().numpy(), color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.show()


        
    

def main():
    feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("data/standard_binary.csv", 800, "Y", n=3, subset=["V6", "V7", "V3"])
    # feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("big_noise_binary.csv", 800, "Y",)
    # feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("big_noise_binary.csv", 800, "Y", n=3)
    
    
    
    # # data = prepare_data("binary_small_noise.csv", 800, "Y", n=3, subset=None)
    # # DiabetesPedigreeFunction
    # # data = prepare_data("diabetes.csv", 600, "Outcome")
    # # feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("binary_distribution_shift.csv", 800, "Y")
    
    
    # print(feats)
    # my_model = LogisticRegression(len(feats))
    # my_model.train(Xtrain, Ytrain)
    # naive = torch.mean(Ytest).item()
    # print("Naive:", max(naive, 1 - naive))
    # print(my_model.evaluate(Xtest, Ytest))
    # # plot(trained_model, data[-2], data[-1])
    # print(f'Accuracy: {eval:.4f}')
    
    
    feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data("old/no_noise.csv", 800, "Y", offset=False)#, n=3)
    model = BoostedDecisionTree(1)
    # model = BaggedDecisionTree(1)
    # model = DecisionTree(1)
    
    

    model.fit(Xtrain, Ytrain)
    print(model.evaluate(Xtest, Ytest))
    # model.train(Xtrain, Ytrain)
    # print(model.evaluate(Xtest, Ytest))
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(Xtrain, Ytrain)
    
    # tree.plot_tree(clf)
    
    # model = NeuralNetwork(len(feats), 3, 64)
    # model.fit(Xtrain, Ytrain)
    # print(model.evaluate(Xtest, Ytest))

if __name__ == "__main__":
    main()