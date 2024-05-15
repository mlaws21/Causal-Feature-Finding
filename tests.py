import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from itertools import combinations
from findsubset import calculate_causal_scores
from model import prepare_data, train, evaluate, LogisticRegression
from graphs import standard_synthetic, diabetes
    
def test_subsets(csvfile, graph_dict, outcome, n=3, num_train=800):

    df = pd.read_csv(csvfile)
    nodes = graph_dict["nodes"]
    edges = graph_dict["edges"]
    
    assert (len(nodes) - 1) >= n
    
    scores = calculate_causal_scores(df, nodes, edges, outcome)
    
    sorted_causers = list(scores.keys())
    ideal_subset = sorted_causers[:n]
    
    all_combinations = list(combinations(sorted_causers, n))

    accuracies = []
    for combo in tqdm(all_combinations):
        
        feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data(csvfile, 800, outcome, n=n, subset=list(combo))
        my_model = LogisticRegression(len(feats))
        trained_model = train(Xtrain, Ytrain, my_model, verbose=False)
        eval = evaluate(trained_model, Xtest, Ytest)
        accuracies.append(eval)
    
    random_accuracy = np.mean(accuracies)
    
    feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data(csvfile, 800, outcome, n=n, subset=ideal_subset)
    my_model = LogisticRegression(len(feats))
    trained_model = train(Xtrain, Ytrain, my_model, verbose=False)
    ideal_accuracy = evaluate(trained_model, Xtest, Ytest)
    
    naive = torch.mean(Ytest).item()
    naive_accuracy = max(naive, 1 - naive)
    
    return round(naive_accuracy, 3), round(random_accuracy, 3), round(ideal_accuracy, 3)
    

def main():
    # for i in range(len(standard_synthetic["nodes"])):
    #     naive, rand, ideal = test_subsets("big_noise_binary.csv", standard_synthetic, "Y", n=i)
    #     print(str(i) + ":")
    #     print("Naive:", naive)
    #     print("Random:", rand)
    #     print("Ideal:", ideal)
        
    for i in range(len(diabetes["nodes"])):
        naive, rand, ideal = test_subsets("diabetes_binary.csv", diabetes, "Outcome", n=i, num_train=600)
        print(str(i) + ":")
        print("Naive:", naive)
        print("Random:", rand)
        print("Ideal:", ideal)
        

        
    
    

if __name__ == "__main__":
    main()