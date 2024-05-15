import random
import pandas as pd
import numpy as np
import sys


def binarize(filename, outname):
    df = pd.read_csv(filename)
    
    df_binary = df.copy()
    for col in df_binary.columns:
        threshold_value = np.mean(df[col])
        df_binary[col] = (df_binary[col] >= threshold_value).astype(int)
    
    df_binary.to_csv(outname, index=False)
    

def main():
    if len(sys.argv) > 2:
        binarize(sys.argv[1], sys.argv[2])
    else:
        print("Usage: TODO")
   

    
if __name__ == "__main__":
    main()