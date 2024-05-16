import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from itertools import combinations
from findsubset import calculate_causal_scores_DAG, calculate_causal_scores_ADMG
from learning import prepare_data, LogisticRegression
from graphs import standard_synthetic, diabetes, mixed_synthetic, upstream_shift, downstream_shift, temp
from generating_functions import standard
from generatedata import generate
from binarize import binarize
from copy import deepcopy
    
def test_subsets(graph_dict, eval_dict=None, n=3, num_train=800, verbose=True):

    outcome = graph_dict["outcome"]
    
    csvfile = graph_dict["data"]
    df = pd.read_csv(csvfile)
    
    if "bi_edges" in graph_dict:
        nodes = graph_dict["nodes"]
        bi_edges = graph_dict["bi_edges"]
        di_edges = graph_dict["di_edges"]
        assert (len(nodes) - 1) >= n
        scores = calculate_causal_scores_ADMG(df, nodes, di_edges, bi_edges, outcome)
        
        
    
    else:
        nodes = graph_dict["nodes"]
        edges = graph_dict["edges"]
        assert (len(nodes) - 1) >= n
        scores = calculate_causal_scores_DAG(df, nodes, edges, outcome)
        
    
    
    
    
    sorted_causers = list(scores.keys())
    ideal_subset = sorted_causers[:n]
    
    all_combinations = list(combinations(sorted_causers, n))

    accuracies = []
    for combo in tqdm(all_combinations, disable=(not verbose)):
        
        feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data(csvfile, num_train, outcome, n=n, subset=list(combo))
        my_model = LogisticRegression(len(feats))
        my_model.fit(Xtrain, Ytrain, verbose=False)
        if eval_dict is not None:
            _, _, _, Xtest, Ytest = prepare_data(eval_dict["data"], num_train, graph_dict["outcome"], n=n, subset=list(combo))
        eval = my_model.evaluate(Xtest, Ytest)
        accuracies.append(eval)
    
    random_accuracy = np.mean(accuracies)
    
    feats, Xtrain, Ytrain, Xtest, Ytest = prepare_data(csvfile, num_train, outcome, n=n, subset=ideal_subset)
    my_model = LogisticRegression(len(feats))
    my_model.fit(Xtrain, Ytrain, verbose=False)
    if eval_dict is not None:
        _, _, _, Xtest, Ytest = prepare_data(eval_dict["data"], num_train, graph_dict["outcome"], n=n, subset=list(combo))
    ideal_accuracy = my_model.evaluate(Xtest, Ytest)
    
    naive = torch.mean(Ytest).item()
    naive_accuracy = max(naive, 1 - naive)
    
    return round(naive_accuracy, 3), round(random_accuracy, 3), round(ideal_accuracy, 3)
    
def run_n_tests(n):
    
    generating_data = standard
    
    starting_names = generating_data["starting_names"]
    starting_generating_boundaries = generating_data["starting_generating_boundaries"] 
    downstream_names = generating_data["downstream_names"]
    downstream_generating_functions = generating_data["downstream_generating_functions"]
    downstream_parents = generating_data["downstream_parents"]
    
    graph = temp
    
    ideal_map = dict((x, []) for x in range(len(graph["nodes"])))
    random_map = dict((x, []) for x in range(len(graph["nodes"])))
    
    accuracies = {"ideal": ideal_map, "random": random_map}
    
    for _ in tqdm(range(n)):
    
        data = generate(starting_names, starting_generating_boundaries, downstream_names, downstream_generating_functions, downstream_parents, 1000)
        df = pd.DataFrame(data)
        df.to_csv(graph["data"], index=False)
        binarize(graph["data"], graph["data"])
        

        for i in range(len(graph["nodes"])):
            naive, rand, ideal = test_subsets(graph, n=i, verbose=False)
            accuracies["ideal"][i].append(ideal)
            accuracies["random"][i].append(rand)
            
    
    out = deepcopy(accuracies)
    for k in accuracies.keys():
        for i in accuracies[k].keys():
            avg = np.mean(accuracies[k][i])
            out[k][i] = avg
            
    return out


def main():
    
    # graph = standard_synthetic
    # eval_graph = None
    # for i in range(len(graph["nodes"])):
    #     naive, rand, ideal = test_subsets(graph, eval_graph,  n=i)
    #     print(str(i) + ":")
    #     print("Naive:", naive)
    #     print("Random:", rand)
    #     print("Ideal:", ideal)
        
    print(run_n_tests(100))

        
    
    

if __name__ == "__main__":
    main()