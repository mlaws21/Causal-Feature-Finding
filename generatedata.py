import random
import pandas as pd
import numpy as np

def generate(starting_names, starting_ranges, downstream_names, downstream_generating_functions, downstream_parents, num_instances):
    lower, upper = starting_ranges
    data = {}
    for i in starting_names:
        data[i] = []
    for i in downstream_names:
        data[i] = []
    
    for _ in range(num_instances):
        for name in starting_names:
            data[name].append(random.uniform(lower, upper))
    
        for name, gen, parents in zip(downstream_names, downstream_generating_functions, downstream_parents):
            
            for p in parents:
                assert len(data[name]) == (len(data[p]) - 1)
            parent_val = (data[x][-1] for x in parents)
            data[name].append(gen(*parent_val))
            
            
        
    
    return data
    
NOISE = 1

V3_GEN = lambda v1, v2: v1 + v2 + random.uniform(-NOISE, NOISE)
V4_GEN = lambda v2, v3: v2 * v3 + random.uniform(-NOISE, NOISE)
V5_GEN = lambda v2, v3: 3*v2 + v3 + random.uniform(-NOISE, NOISE)
Y_GEN = lambda v1, v3, v4, v6: v1 * 0.4 + v3*v6*0.3 + v4*0.1 + v6 + random.uniform(-NOISE, NOISE)
V7_GEN = lambda y: y*y*0.5 + random.uniform(-NOISE, NOISE)
V8_GEN = lambda v2, y: v2 + y*0.2 + random.uniform(-NOISE, NOISE)
# V8_GEN = lambda v2, y: v2 + y*0.2 + random.uniform(-NOISE, NOISE) + 5
V9_GEN = lambda v7, v8: v7*0.4 + v8*v7*0.1 + random.uniform(-NOISE, NOISE)



def main():
    starting_names = ["V1", "V2", "V6"]
    downstream_names = ["V3", "V4", "V5", "Y", "V7", "V8", "V9"]
    downstream_generating_functions = [V3_GEN, V4_GEN, V5_GEN, Y_GEN, V7_GEN, V8_GEN, V9_GEN]
    downstream_parents = [("V1", "V2"), ("V2", "V3"), ("V2", "V3"), ("V1", "V3", "V4", "V6"), ("Y"), ("V2", "Y"), ("V7", "V8")]
    data = generate(starting_names, (0, 1), downstream_names, downstream_generating_functions, downstream_parents, 1000)
    # print(data)
    
    df = pd.DataFrame(data)
    df.to_csv("big_noise.csv", index=False)
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