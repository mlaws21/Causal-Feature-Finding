import random
import pandas as pd
import numpy as np
from generating_functions import standard, downstream_shift, mixed_standard, upstream_shift

def generate(starting_names, starting_generating_boundaries, downstream_names, downstream_generating_functions, downstream_parents, num_instances):
    data = {}
    for i in starting_names:
        data[i] = []
    for i in downstream_names:
        data[i] = []
    
    for _ in range(num_instances):
        for name, starting_range in zip(starting_names, starting_generating_boundaries):
            lower, upper = starting_range
            data[name].append(random.uniform(lower, upper))
    
        for name, gen, parents in zip(downstream_names, downstream_generating_functions, downstream_parents):
            
            for p in parents:
                assert len(data[name]) == (len(data[p]) - 1)
            parent_val = (data[x][-1] for x in parents)
            data[name].append(gen(*parent_val))
            
            
        
    
    return data



def main():
    
    generating_data = downstream_shift
    
    starting_names = generating_data["starting_names"]
    starting_generating_boundaries = generating_data["starting_generating_boundaries"] 
    downstream_names = generating_data["downstream_names"]
    downstream_generating_functions = generating_data["downstream_generating_functions"]
    downstream_parents = generating_data["downstream_parents"]
    data = generate(starting_names, starting_generating_boundaries, downstream_names, downstream_generating_functions, downstream_parents, 1000)
    # print(data)
    
    df = pd.DataFrame(data)
    df.to_csv(generating_data["name"], index=False)
    
    
    
    # starting_names = ["A", "U"]
    # starting_generating_boundaries = [(0, 1), (0, 1)]
    # downstream_names = ["Y"]
    # downstream_generating_functions = [lambda a, u: a + u + random.uniform(-.3, .3)]
    # downstream_parents = [("A", "U")]
    # data = generate(starting_names, starting_generating_boundaries, downstream_names, downstream_generating_functions, downstream_parents, 1000)
    
    
    # df = pd.DataFrame(data)
    # df.to_csv("quick.csv", index=False)
    # # df = df.round(2)
    # # print(np.mean(df["Y"]))
    # # print(df["Y"])
    # threshold_value = np.mean(df["Y"])
    # df_binary_out = df.copy()
    # df_binary_out["Y"] = (df_binary_out["Y"] >= threshold_value).astype(int)
    
    # # df_binary_out.to_csv("synth_binary_out.csv", index=False)

    # # df.to_csv("synth.csv", index=False)
    
    # df_binary = df.copy()
    # for col in df_binary.columns:
    #     threshold_value = np.mean(df[col])
    #     df_binary[col] = (df_binary[col] >= threshold_value).astype(int)
    
    # df_binary.to_csv("binary_small_noise.csv", index=False)
    

    
if __name__ == "__main__":
    main()