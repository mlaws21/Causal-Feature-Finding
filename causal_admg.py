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
        Returns a list of names of the siblings
        """
        return [s.name for s in self.vertices[vertex_name].siblings]
    
    def get_all_siblings(self, vertex_name: str) -> list[str]:
        """
        Returns a list of names of all siblings (all nodes connected to [vertex_name] by bidirected edges and siblings
        of siblings etc)
        """
        q = deque()
        
        # first_sibs = self.get_siblings(vertex_name)
        siblings = set()
        
        q.append(vertex_name)
        
        while len(q) > 0:
            curr_node = q.popleft()
            curr_siblings = self.get_siblings(curr_node)
            for i in curr_siblings:
                if i not in siblings:
                    siblings.add(i)
                    q.append(i)
        
        
        siblings.discard(vertex_name)
        
        return siblings
        

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
        for edge in self.di_edges:
            if edge[0] != a_name:
                edges.append(edge)

        G_Inter = CausalADMG(self.vertices, edges, self.bi_edges)
        return G_Inter.m_separated(a_name, y_name, z_names)

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
    
    def find_valid_adjustment_set(self, treatment, outcome):
        valid = set()
        causer = treatment
        causee = outcome
        if outcome not in self.get_descendants(treatment):
            causer, causee = causee, causer
            
        if causee not in self.get_descendants(causer):
            # no causal relationship
            return None
            
        parents = self.get_parents(causer)
        for i in parents:
            valid.add(i)
            
        # print(valid)
            
        siblings = self.get_all_siblings(causer)
        # print("sibs", siblings)
        for sib in siblings:
            valid.add(sib)

            for pa in self.get_parents(sib):
                valid.add(pa)
                
                
        return causer, causee, valid
        
    def find_minimal_adjustment_set(self, treatment, outcome):
        valid_output = self.find_valid_adjustment_set(treatment, outcome)
        if valid_output is None:
            return None
        causer, causee, optimal = valid_output
        # print("optimal", optimal)
        # this is a naive approach
        min_val = set(optimal)
        
        all_subsets = list(chain.from_iterable(combinations(optimal, r) for r in range(len(optimal)+1)))

        for i in all_subsets:
            si = set(i)
            if self.valid_backdoor_set(causer, causee, si):
                # print(si)
                if len(si) < len(min_val):
                    min_val = si
                    # TODO i think we can break here because it is in order
                
        return causer, causee, list(min_val)
    
    def directed_dfs(self, node, target, visited, path, paths):
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
                self.directed_dfs(child, target, visited, path, paths)
        
        path.pop()  # Remove the current node from the path
        visited.remove(node) 
    
    def find_causal_paths(self, treatment, outcome):
        visited = set()
        path = []
        paths = []
        self.directed_dfs(treatment, outcome, visited, path, paths)
        return paths
        
        
    def find_instrument(self, treatment, outcome):
        #TODO should we run an instrument test?
        No_umeasured_confouding = CausalDAG(self.vertices, self.di_edges)
        
        candidates = [x for x in self.vertices if (x != treatment and x != outcome)]
        
        # need to find some set C that is a valid adjustment set for A -> Y and Z -> A
        
        # two ways find a set for the first and second and combine (this is not gaurenteeded to always
        # find the set if there is one) or iterate over all the sets
        
        for instrument in candidates:
            
            valid_C = None
            possible_confounders = [x for x in candidates if x != instrument]
            all_subsets = list(chain.from_iterable(combinations(possible_confounders, r) for r in range(len(possible_confounders)+1)))
            
            for i in all_subsets:
                si = set(i)
                if self.valid_backdoor_set(instrument, outcome, si) and self.valid_backdoor_set(instrument, treatment, si):
                    valid_C = si
                    break # in order of smallest to biggest so we can stop early
            
            
            if valid_C is None:
                # there is no valid adjustment set C
                continue
            
            
            instrument_causes_treatment = not No_umeasured_confouding.d_separated(instrument, treatment, valid_C)
            
            # if No_umeasured_confouding.get(treatment, outcome, valid_C)
            
            if not instrument_causes_treatment:
                continue
            
            return instrument, valid_C
            
        return None
        
    
    def find_mediator(self, treatment, outcome):
        
        causal_paths = self.find_causal_paths(treatment, outcome)
        candidates = [x for x in self.vertices if (x != treatment and x != outcome)]
        valid_mediators = set(candidates)
        for path in causal_paths:
            valid_mediators &= set(path)
        
        # valid_C = None
        for med in valid_mediators:
            possible_confounders = [x for x in candidates if x != med]
            all_subsets = list(chain.from_iterable(combinations(possible_confounders, r) for r in range(len(possible_confounders)+1)))
            for i in all_subsets:
                si = set(i)

                if self.valid_backdoor_set(treatment, med, si) and self.valid_backdoor_set(med, outcome, si.union({treatment})):
                    # valid_C = si

                    return med, si
                    

        return None