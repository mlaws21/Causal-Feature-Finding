from causal import CausalDAG, augmented_ipw, backdoor_adjustment, ipw ,compute_confidence_intervals
import pandas as pd
from collections import OrderedDict


#TODO mess with graph

def calculate_causal_scores(data: pd.DataFrame, nodes: list[str], edges: list[(str, str)], outcome: str, estimator=augmented_ipw, l=0.5):
    causal_score = {}
    G = CausalDAG(nodes, edges)
    treatment_nodes = [x for x in nodes if x != outcome]
    
    for node in treatment_nodes:
        minop_set = G.find_minimal_optimal_backdoor(node, outcome)
        if minop_set is None:
            causal_score[node] = 0.0

        else:
            causer, causee, adj_set = minop_set
            causal_effect = estimator(data, causer, causee, adj_set)

            # TODO confidence intervals?
            if causee != outcome:
                # penalize downstream causation
                causal_effect *= l
            
            causal_score[node] = abs(causal_effect)
    
    ordered_score = OrderedDict(sorted(causal_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    return ordered_score
            
        
    

def main():
    df = pd.read_csv('big_noise_binary.csv')
    nodes = ['V8', 'V2', 'V4', 'Y', 'V6', 'V3', 'V5', 'V7', 'V9', 'V1']
    edges = [('V2', 'V8'), ('V2', 'V4'), ('Y', 'V8'), ('V4', 'Y'), ('V2', 'V3'), ('V3', 'Y'), ('V2', 'V5'), ('V3', 'V5'), ('Y', 'V7'), ('V8', 'V9'), ('V7', 'V9'), ('V6', 'Y'), ('V1', 'V3'), ('V1', 'Y'), ('V3', 'V4')]
    scores = calculate_causal_scores(df, nodes, edges, "Y")

    
    # df = pd.read_csv("diabetes_binary.csv")
    # nodes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    # edges = [('Age', 'BloodPressure'), ('Age', 'Glucose'), ('Age', 'Outcome'), ('Age', 'Pregnancies'), ('BMI', 'BloodPressure'), ('BMI', 'Insulin'), ('BMI', 'Outcome'), ('BloodPressure', 'Glucose'), ('DiabetesPedigreeFunction', 'Insulin'), ('DiabetesPedigreeFunction', 'Outcome'), ('DiabetesPedigreeFunction', 'SkinThickness'), ('Glucose', 'Insulin'), ('Glucose', 'Outcome'), ('SkinThickness', 'BMI'), ('SkinThickness', 'Insulin')]
    # scores = calculate_causal_scores(df, nodes, edges, "Outcome")
    
    # df = pd.read_csv("spotify_binary.csv")
    # nodes = ['track_popularity', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'tempo', 'duration_ms']
    # edges = [('acousticness', 'tempo'), ('acousticness', 'track_popularity'), ('duration_ms', 'acousticness'), ('duration_ms', 'instrumentalness'), ('duration_ms', 'speechiness'), ('duration_ms', 'track_popularity'), ('instrumentalness', 'acousticness'), ('instrumentalness', 'speechiness'), ('instrumentalness', 'tempo'), ('instrumentalness', 'track_popularity'), ('key', 'speechiness'), ('key', 'tempo'), ('loudness', 'acousticness'), ('loudness', 'duration_ms'), ('loudness', 'instrumentalness'), ('loudness', 'speechiness'), ('loudness', 'tempo'), ('loudness', 'track_popularity'), ('speechiness', 'acousticness'), ('tempo', 'track_popularity')]
    # scores = calculate_causal_scores(df, nodes, edges, "track_popularity")

    
    print(scores)


if __name__ == "__main__":
    main()