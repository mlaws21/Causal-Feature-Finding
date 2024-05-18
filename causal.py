from graphviz import Digraph
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from collections import deque
from itertools import chain, combinations
from collections.abc import Callable


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

class CausalDAG:
    """
    DAG class
    """

    def __init__(self, vertex_names: list[str], edges: list[(str, str)]) -> None:
        """
        Constructor for the causal DAG class
        """

        self.vertices = {v: Vertex(v) for v in vertex_names} # dictionary mapping vertex names to Vertex objects
        self.edges = [] # list of tuples corresponding to edges in the DAG

        # loop over and initialize all vertices to have parent-child relations specified by the edges
        for parent_name, child_name in edges:
            self.edges.append((parent_name, child_name))
            # get the corresponding vertex objects
            parent_vertex = self.vertices.get(parent_name)
            child_vertex = self.vertices.get(child_name)
            # add to the parent/child sets
            parent_vertex.children.add(child_vertex)
            child_vertex.parents.add(parent_vertex)


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

    def get_descendants(self, vertex_name: str) -> list[str]:
        """
        Returns a list of strings corresponding to descendants of the given vertex.
        Note by convention, the descendants of a vertex include the vertex itself.
        """
        # implement this

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

    def d_separated(self, x_name: str, y_name: str, z_names: list[str]) -> bool:
        """
        Check if X _||_ Y | Z using d-separation
        """

        stack = [(x_name, "up")] # stack for vertices to be explored next
        visited = set() # set of (vertex, direction) pairs that have already been explored

        while len(stack) > 0:

            v_name, direction = stack.pop()

            if (v_name, direction) in visited:
                continue

            # we reached Y through an open path so return False
            if v_name == y_name:
                return False

            visited.add((v_name, direction))

            # cases for active forks and chain
            if direction == "up" and v_name not in z_names:

                for child in self.get_children(v_name):
                    stack.append((child, "down"))
                for parent in self.get_parents(v_name):
                    stack.append((parent, "up"))

            # cases for active chain and colliders
            elif direction == "down":

                if v_name not in z_names:
                    for child in self.get_children(v_name):
                        stack.append((child, "down"))

                if len(set(self.get_descendants(v_name)).intersection(z_names)) != 0:
                    for parent in self.get_parents(v_name):
                        stack.append((parent, "up"))

        return True

    def valid_backdoor_set(self, a_name: str, y_name: str, z_names: list[str]) -> bool:
        """
        Check if Z is a valid backdoor set for computing the effect of A on Y
        """

        # check the descendants criterion
        descendants_a = self.get_descendants(a_name)
        if len(set(descendants_a).intersection(z_names)) != 0:
            return False

        # check d-sep criterion in graph where we remove outgoing edges A->o
        edges = []
        for edge in self.edges:
            if edge[0] != a_name:
                edges.append(edge)

        G_Abar = CausalDAG(self.vertices, edges)
        return G_Abar.d_separated(a_name, y_name, z_names)

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

        for parent, child in self.edges:
            dot.edge(parent, child, color="blue")

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
        
class Bi_Vertex:
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

        self.vertices = {v: Bi_Vertex(v) for v in vertex_names} # dictionary mapping vertex names to Vertex objects
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
            
            return instrument, list(valid_C)
            
        return None
        
    
    def find_mediator(self, treatment, outcome):
        # currently only find single mediators
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

                    return med, list(si)
                    

        return None
    
def squash_paths(paths):
    out = set()
    for p in paths:
        for e in p:
            out.add(e)
            
    return out

def backdoor_adjustment(data: pd.DataFrame, a_name: str, y_name: str, c_names: list[str]) -> float:
    """
    Perform backdoor adjustment for a given treatment A and outcome Y using
    the covariates in Z
    """

    # make a regression formula
    c_names = ["1"] + c_names
    c_formula = " + ".join(c_names)
    regression_formula = f"{y_name} ~ {c_formula} + {a_name}"

    # fit a regression depending on whether Y is binary or not
    if set(data[y_name]) == {0, 1}:
        model = smf.glm(formula=regression_formula, family=sm.families.Binomial(), data=data).fit()
    else:
        model = smf.glm(formula=regression_formula, family=sm.families.Gaussian(), data=data).fit()

    data_a1 = data.copy() # make a copy for the interventional datasets
    data_a1[a_name] = 1
    data_a0 = data.copy()
    data_a0[a_name] = 0

    return round(np.mean(model.predict(data_a1) - model.predict(data_a0)), 3)

def ipw(data: pd.DataFrame, a_name: str, y_name: str, c_names: list[str]) -> float:
    """
    Perform IPW for a given treatment A and outcome Y using
    the covariates in Z
    """

    # make a regression formula
    c_names = ["1"] + c_names
    c_formula = " + ".join(c_names)
    regression_formula = f"{a_name} ~ {c_formula}"

    # fit a logistic regression and get propensity scores
    model = smf.glm(formula=regression_formula, family=sm.families.Binomial(), data=data).fit()
    p_scores = model.predict(data)

    # return the result of the IPW computation
    return round(np.mean(data[a_name]/p_scores*data[y_name] - (1-data[a_name])/(1-p_scores)*data[y_name]), 3)


def augmented_ipw(data: pd.DataFrame, a_name: str, y_name: str, z_names: list[str]) -> float:
    """
    Perform AIPW for a given treatment A and outcome Y using
    the covariates in Z
    """

    # make regression formulas
    z_names = ["1"] + z_names
    z_formula = " + ".join(z_names)
    formula_a = f"{a_name} ~ {z_formula}"
    formula_y = f"{y_name} ~ {z_formula} + {a_name}"

    # fit regression models
    model_a = sm.GLM.from_formula(formula=formula_a, data=data, family=sm.families.Binomial()).fit()
    if set(data[y_name]) == {0, 1}:
        model_y = smf.glm(formula=formula_y, family=sm.families.Binomial(), data=data).fit()
    else:
        model_y = smf.glm(formula=formula_y, family=sm.families.Gaussian(), data=data).fit()


    # make interventional datasets
    data_a0 = data.copy()
    data_a0[a_name] = 0
    data_a1 = data.copy()
    data_a1[a_name] = 1

    # get predictions for Y, Ya1, Ya0 and p(A|Z)
    Yhat = model_y.predict(data)
    Ya0 = model_y.predict(data_a0)
    Ya1 = model_y.predict(data_a1)
    p_scores = model_a.predict(data)

    # compute and return AIPW results
    aipw_a1 = np.mean(data[a_name]/p_scores*(data[y_name]-Yhat) + Ya1)
    aipw_a0 = np.mean((1-data[a_name])/(1-p_scores)*(data[y_name]-Yhat) + Ya0)
    return round(aipw_a1 - aipw_a0, 3)

def iv_adjustment(data: pd.DataFrame, a_name: str, y_name: str, iv_name: str, c_names: list[str]) -> float:
    top = backdoor_adjustment(data, iv_name, y_name, c_names)
    bottom = backdoor_adjustment(data, iv_name, a_name, c_names)
    result = top / bottom
    return round(result, 3)
    
    

def frontdoor_ipw(data: pd.DataFrame, a_name: str, y_name: str, m_name: str, c_names: list[str]) -> float:
    """
    Perform front door adjustment for a given treatment A and outcome Y using
    the mediator M and the covariates in C
    """
    # make a regression formula for M ~ A + C
    c_names = ["1"] + c_names
    c_formula = " + ".join(c_names)
    regression_formula = f"{m_name} ~ {a_name} + {c_formula}"

    # fit a logistic regression for M
    model = smf.glm(formula=regression_formula, family=sm.families.Binomial(), data=data).fit()

    # make interventional datasets
    data_a1 = data.copy(); data_a0 = data.copy()
    data_a1[a_name] = 1
    data_a0[a_name] = 0

    p_MA = model.predict(data)
    p_Ma1 = model.predict(data_a1)
    p_Ma0 = model.predict(data_a0)

    result = np.mean((p_Ma1 / p_MA) * data[y_name]) -  np.mean((p_Ma0 / p_MA) * (data[y_name]))
    
    return round(result, 3)

    
def compute_confidence_intervals(data: pd.DataFrame, estimator_args: tuple, estimator: Callable,
                                 num_bootstraps: int=200, alpha: float=0.05, ) -> tuple[float, float]:
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    for i in range(num_bootstraps):

        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)

        # add estimate from resampled data
        estimates.append(estimator(data_sampled, *estimator_args))

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]

    return round(q_low, 3), round(q_up, 3)

