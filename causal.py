from graphviz import Digraph
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from collections import deque
from itertools import chain, combinations


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

def compute_confidence_intervals(data: pd.DataFrame, a_name: str, y_name: str, z_names: list[str], estimator, 
                                 num_bootstraps: int=200, alpha: float=0.05) -> tuple[float, float]:
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
        estimates.append(estimator(data_sampled, a_name, y_name, z_names))

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]

    return round(q_low, 3), round(q_up, 3)

