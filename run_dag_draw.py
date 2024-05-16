from dag_drawer import Dag_Drawer
import sys 
import ast

def main():
    
    nodes = []
    edges = []
    
    if len(sys.argv) > 1:
        nodes = ast.literal_eval(sys.argv[1])

    if len(sys.argv) > 2:
        edges = ast.literal_eval(sys.argv[2])

    Dag_Drawer(nodes, edges, term=True)
    
if __name__ == "__main__":
    main()
