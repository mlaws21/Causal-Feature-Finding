from graphviz import Digraph
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from collections import deque
from itertools import chain, combinations

from causal import CausalDAG, squash_paths

class Vertex:
    """
    Vertex class
    """

    def __init__(self, name: str):
        """
        Constructor for the Vertex class
        """
        self.name = name
        self.parents = set() # set consisting of Vertex objects that are parents of this vertex
        self.children = set() # set consisting of Vertex objects that are children of this vertex
        self.siblings = set() # set consisting of Vertex objects that are siblings of this vertex
        


class CausalADMG:
    """
    ADMG class
    """

    def __init__(self, vertex_names: list[str], di_edges: list[(str, str)],  bi_edges: list[(str, str)]) -> None:
        """
        Constructor for the causal DAG class
        """

        self.vertices = {v: Vertex(v) for v in vertex_names} # dictionary mapping vertex names to Vertex objects
        self.di_edges = [] # list of tuples corresponding to edges in the DAG
        self.bi_edges = [] # list of tuples corresponding to edges in the DAG
        

        # loop over and initialize all vertices to have parent-child relations specified by the edges
        for parent_name, child_name in di_edges:
            self.di_edges.append((parent_name, child_name))
            # get the corresponding vertex objects
            parent_vertex = self.vertices.get(parent_name)
            child_vertex = self.vertices.get(child_name)
            # add to the parent/child sets
            parent_vertex.children.add(child_vertex)
            child_vertex.parents.add(parent_vertex)

        for brother, sister in bi_edges:
            self.bi_edges.append((brother, sister))
            # get the corresponding vertex objects
            brother_vertex = self.vertices.get(brother)
            sister_vertex = self.vertices.get(sister)
            # add to the parent/child sets
            brother_vertex.siblings.add(sister_vertex)
            sister_vertex.siblings.add(brother_vertex)


    def get_parents(self, vertex_name: str) -> list[str]:
        """
        Returns a list of names of the parents
        """
        return [p.name for p in self.vertices[vertex_name].parents]

    def get_children(self, vertex_name: str) -> list[str]:
        """
        Returns a list of names of the parents
        """
        return [c.name for c in self.vertices[vertex_name].children]
    
    def get_siblings(self, vertex_name: str) -> list[str]:
        """
        Returns a list of names of the parents
        """
        return [s.name for s in self.vertices[vertex_name].siblings]

    def get_descendants(self, vertex_name: str) -> list[str]:
        """
        Returns a list of strings corresponding to descendants of the given vertex.
        Note by convention, the descendants of a vertex include the vertex itself.
        """

        q = deque()
        q.append(vertex_name)
        decendents = {vertex_name}

        while len(q) > 0:
          curr = q.popleft()
          children = self.get_children(curr)
          for i in children:
            if i not in decendents:
              decendents.add(i)
              q.append(i)


        return decendents

    def m_separated(self, x_name: str, y_name: str, z_names: list[str]) -> bool:
        """
        Check if X _||_ Y | Z using m-separation
        This is a bit of a cheeky way to to this but basically we will compute the inverse 
        of the latent projection operator (or at least an equivalent graph) then we will
        just run a d-sep algorithm
        """
        #"U" + str(x) for x in range(len(self.bi_edges))
        unobserved_nodes = []
        unobserved_edges = []
        for i, ele in enumerate(self.bi_edges):
            brother, sister = ele
            un_node = "U" + str(i)
            unobserved_edges.append((un_node, brother))
            unobserved_edges.append((un_node, sister))
            unobserved_nodes.append(un_node)
            
        inverse_latent_edges = self.di_edges + unobserved_edges
        inverse_latent_nodes = list(self.vertices.keys()) + unobserved_nodes
        
        inverse_latent_DAG = CausalDAG(inverse_latent_nodes, inverse_latent_edges)
        
        return inverse_latent_DAG.d_separated(x_name, y_name, z_names)
        
        


    def valid_backdoor_set(self, a_name: str, y_name: str, z_names: list[str]) -> bool:
        """
        Check if Z is a valid backdoor set for computing the effect of A on Y
        """

        # check the descendants criterion
        descendants_a = self.get_descendants(a_name)
        if len(set(descendants_a).intersection(z_names)) != 0:
            return False

        # check m-sep criterion in graph where we remove outgoing edges A->o
        edges = []
        for edge in self.edges:
            if edge[0] != a_name:
                edges.append(edge)

        G_Inter = CausalADMG(self.vertices, edges, self.bi_edges)
        return G_Inter.d_separated(a_name, y_name, z_names)

    def draw(self):
        """
        Method for visualizaing the DAG
        """

        dot = Digraph()
        dot.graph_attr["rankdir"] = "LR"

        for v_name in self.vertices:
            dot.node(
                v_name,
                shape="plaintext",
                height=".5",
                width=".5",
            )

        for parent, child in self.di_edges:
            dot.edge(parent, child, color="blue")
            
        for brother, sister in self.bi_edges:
            dot.edge(brother, sister, color="red", dir="both")  

        return dot
    
    
    def dfs(self, node, target, visited, path, paths):
        if node == target:
            path.append(node)
            toadd = list(path) # the list() command copies it -- you get a nasty error without it
            paths.append(toadd[1:]) 
            path.pop() 
            return
        
        visited.add(node)
        path.append(node)
        for child in self.get_children(node):
            if child not in visited:
                self.dfs(child, target, visited, path, paths)
        
        path.pop()  # Remove the current node from the path
        visited.remove(node) 
    
    def find_causal_paths(self, treatment, outcome):
        visited = set()
        path = []
        paths = []
        self.dfs(treatment, outcome, visited, path, paths)
        return paths
        
    
    
    def find_optimal_backdoor(self, treatment, outcome):
        # if self.d_separated(treatment, )
        
        causer = treatment
        causee = outcome
        
        path = squash_paths(self.find_causal_paths(causer, causee))

        if len(path) == 0:
            # flip the effect
            causer = outcome
            causee = treatment
            path = squash_paths(self.find_causal_paths(causer, causee))
            
        if len(path) == 0:
            return None
        
        optimal = set()
        for c in path:
            for p in self.get_parents(c):
                if p not in path and p != causer and p != causee:
                    optimal.add(p)
                    
        return causer, causee, optimal
        
    def find_minimal_optimal_backdoor(self, treatment, outcome):
        optimal_output = self.find_optimal_backdoor(treatment, outcome)
        if optimal_output is None:
            return None
        causer, causee, optimal = optimal_output
        # print("optimal", optimal)
        # this is a naive approach
        min_op = set(optimal)
        
        all_subsets = list(chain.from_iterable(combinations(optimal, r) for r in range(len(optimal)+1)))

        for i in all_subsets:
            si = set(i)
            if self.valid_backdoor_set(causer, causee, si):
                # print(si)
                if len(si) < len(min_op):
                    min_op = si
                
        return causer, causee, list(min_op)
        
        
        
     