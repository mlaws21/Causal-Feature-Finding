import jpype.imports
# try:
#     jpype.startJVM(classpath=[f"resources/tetrad-current.jar"])
# except OSError:
#     print("JVM already started", OSError.filename2())
#     # return

import pandas as pd
import tetrad.TetradSearch as ts
from graphviz import Digraph


# TODO add negative numbers
def add_tier_knowledge(tet_search, knowlegefile, cols=None):
    f = open(knowlegefile, "r")
    for line in f:
        sline = line.strip()
        if sline != "":
            tier_data = sline.split(" ")
            try:
                tier_num = int(tier_data[0])
            except:
                "ERROR: Bad knowledge format"
                return
                
            for i in tier_data[1:]:
                tet_search.add_to_tier(tier_num, i)
            
def parseEdge(e):
    spl = e.split(" ")
    return (spl[1], spl[2], spl[3])
   
def parsePCout(out):
    nodes_next = False
    edges_next = False
    edges = []
    lines = out.split("\n")
    for l in lines:
        l = l.strip()
        
        if l == "":
            edges_next = False
        
        if nodes_next:
            nodes = l.split(";")
            nodes_next = False
            continue
            
        if edges_next:
            edges.append(parseEdge(l))
            continue
            
        if l == "Graph Nodes:":
            nodes_next = True
        elif l == "Graph Edges:":
            edges_next = True
        else:
            # random line
            pass
    
    return list(nodes), list(edges)
        
# calls tetrad PC
# returns tetrad PC search string
def run_pc(datafile, discrete=True, knowledge=None):


    data = pd.read_csv(datafile)
    if not discrete:
        data = data.astype({col: "float64" for col in data.columns})
    
    search = ts.TetradSearch(data)
    search.set_verbose(False)

    if discrete:
        search.use_bdeu(sample_prior=10, structure_prior=0)
        search.use_chi_square(alpha=0.1)
    else:
        search.use_sem_bic()
        search.use_fisher_z(alpha=0.05)
    
    if knowledge is not None:
        add_tier_knowledge(search, knowledge)

    search.run_pc()
    return parsePCout(search.get_string())

def run_fci(datafile, discrete=True, knowledge=None):


    data = pd.read_csv(datafile)
    if not discrete:
        data = data.astype({col: "float64" for col in data.columns})
    
    search = ts.TetradSearch(data)
    search.set_verbose(False)

    if discrete:
        search.use_bdeu(sample_prior=10, structure_prior=0)
        search.use_chi_square(alpha=0.1)
    else:
        search.use_sem_bic()
        search.use_fisher_z(alpha=0.05)
    
    if knowledge is not None:
        add_tier_knowledge(search, knowledge)
        

    search.run_fci()
    return parsePCout(search.get_string())

def draw(nodes, edges):
    """
    Method for visualizaing the DAG
    """

    dot = Digraph()
    dot.graph_attr["rankdir"] = "LR"

    for v_name in nodes:
        
        dot.node(
            str(v_name),
            shape="plaintext",
            height=".5",
            width=".5",
        )

    for parent, direction, child in edges:
        if direction == "-->":
            dot.edge(str(parent), str(child), color="blue")
        elif direction == "---":
            dot.edge(str(parent), str(child), color="brown", dir="none")
        
        elif direction == "<->":
            dot.edge(str(parent), str(child), color="red", dir="both")
            
        elif direction == "o-o":
            dot.edge(str(parent), str(child), color="orange", dir="none")
            
        elif direction == "o->":
            dot.edge(str(parent), str(child), color="green")
        
        else:
            print(f"ERROR: bad edge type: {direction}")
            

    return dot

def main():
    nodes, edges = run_pc("binary_out.csv")

    print(nodes)
    print(edges)
    
if __name__ == "__main__":
    main()