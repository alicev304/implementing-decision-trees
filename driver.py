# driver file
# to implement and test decision tree algorithms
# developer and owner of this program: Shrey S V (ssv170001)
# created by alice_v3.0.4 on 02/01/2019
# last modified on 02/02/2019

import os
import sys
import pandas as pd
import numpy as np
import copy as cp
import random

L = int(sys.argv[1])
K = int(sys.argv[2])
rel_training_set_path = sys.argv[3]
rel_testing_set_path = sys.argv[4]
rel_validation_set_path = sys.argv[5]
to_print = True if sys.argv[6] == "yes" else False

# L = 10
# K = 10
# rel_training_set_path = "data_set2/training_set.csv"
# rel_testing_set_path = "data_set2/test_set.csv"
# rel_validation_set_path = "data_set2/validation_set.csv"
# to_print = False

class Node():
    
    def __init__(self, attr = ""):
        self._attribute = attr
        self._child0 = None
        self._child1 = None
        self._tvalue = None
        self._count0 = None
        self._count1 = None
        self._type = None

script_dir = os.path.dirname(__file__)
abs_training_set_path = os.path.join(script_dir, rel_training_set_path)
abs_testing_set_path = os.path.join(script_dir, rel_testing_set_path)
abs_validation_set_path = os.path.join(script_dir, rel_validation_set_path)

training_set = pd.read_csv(abs_training_set_path)
testing_set = pd.read_csv(abs_testing_set_path)
validation_set = pd.read_csv(abs_validation_set_path)

attribute = list(training_set)
target = attribute[-1]
attribute = attribute[: -1]
n = training_set.shape[0]
node_list = []

def log2(x):
    if x: return np.log2(x)
    else: return 0

def shannon_entropy(S):
    P0 = len(S[S[target] == 0]) / len(S) if len(S) else 0
    P1 = len(S[S[target] == 1]) / len(S) if len(S) else 0
    return - P0 * log2(P0) - P1 * log2(P1)

def conditional_entropy(S, X):
    SX0 = S[S[X] == 0]
    SX1 = S[S[X] == 1]
    return shannon_entropy(S) - shannon_entropy(SX0) * len(SX0) / len(S) \
        - shannon_entropy(SX1) * len(SX1) / len(S) if len(S) else 0

def variance_impurity(S):
    P0 = len(S[S[target] == 0]) / len(S) if len(S) else 0
    P1 = len(S[S[target] == 1]) / len(S) if len(S) else 0
    return (P0 * P1) if len(S) else 0

def conditional_impurity(S, X):
    SX0 = S[S[X] == 0]
    SX1 = S[S[X] == 1]
    return variance_impurity(S) - variance_impurity(SX0) * len(SX0) / len(S) \
        - variance_impurity(SX1) * len(SX1) / len(S) if len(S) else 0

def gain(S, X, heuristic):
    if heuristic == "entropy": return conditional_entropy(S, X)
    if heuristic == "impurity": return conditional_impurity(S, X)
    else:
        print("invalid heuristic")
        return -1

def build_DT(root, S, attributes, c0, c1, s = "entropy"):
    if len(S[S[target] == 0]) == len(S):
        root = Node(target)
        root._count0 = len(S[S[target] == 0])
        root._count1 = len(S[S[target] == 1])
        root._tvalue = 0
        return root
    if len(S[S[target] == 1]) == len(S):
        root = Node(target)
        root._count1 = len(S[S[target] == 0])
        root._count0 = len(S[S[target] == 1])
        root._tvalue = 1
        return root
    if not attributes:
        root = Node(target)
        root._count0 = len(S[S[target] == 0])
        root._count1 = len(S[S[target] == 1])
        root._tvalue = 0 if root._count0 > root._count1 else 1
        return root
    max_gain_attr = ""
    max_gain = -1
    for X in attributes:
        gainX = gain(S, X, s)
        if gainX > max_gain:
            max_gain = gainX
            max_gain_attr = X
    root = Node(max_gain_attr)
    root._count0 = c0
    root._count1 = c1
    S0 = S[S[max_gain_attr] == 0]
    S1 = S[S[max_gain_attr] == 1]
    attribute_list = attributes[:]
    attribute_list.remove(max_gain_attr)
    root._child0 = build_DT(root._child0, S0, attribute_list, 
                                len(S[(S[max_gain_attr] == 0) & S[target] == 0]), len(S[(S[max_gain_attr] == 0) & S[target] == 1]))
    root._child1 = build_DT(root._child1, S1, attribute_list, 
                                len(S[(S[max_gain_attr] == 1) & S[target] == 0]), len(S[(S[max_gain_attr] == 1) & S[target] == 1]))
    return root
    
def print_DT(root, depth = 0):
    tab = "    "
    if not root: return
    if root._attribute == target:
        print(tab * depth + target + ": " + str(root._tvalue))
    if root._child0:
        print(tab * depth + root._attribute + ": 0")
        print_DT(root._child0, depth + 1)
    if root._child1:
        print(tab * depth + root._attribute + ": 1")
        print_DT(root._child1, depth + 1)
    return

def validate_config(root, config):
    if root._attribute == target: return root._tvalue
    if config[root._attribute]:
        return validate_config(root._child1, config)
    return validate_config(root._child0, config)

def validate_DT(root, S):
    par = 0
    for item, configuration in S.iterrows():
        prediction = validate_config(root, configuration)
        if prediction == configuration[target]: par += 1
    return par * 100.0 / len(S)

def create_NL(node):
    global node_list
    if node._attribute != target:
        node_list.append(node)
        create_NL(node._child0)
        create_NL(node._child1)
    return
    
def updateNL(node):
    global node_list
    if node and node._attribute != target:
        node_list.remove(node)
        updateNL(node._child0)
        updateNL(node._child1)        

def prune_DT(root, P):
    global node_list
    node = node_list[P]
    if node:
        updateNL(node._child0)
        updateNL(node._child1)
        node._child0 = Node(target)
        node._child1 = Node(target)
        node._child0._tvalue = node._count0
        node._child1._tvalue = node._count1
    return root

def algorithm_1(L, K):
    global node_list
    root = None
    root = build_DT(root, training_set, attribute, -1, -1)
    root_best = cp.deepcopy(root)
    print(validate_DT(root, validation_set))
    for i in range(L):
        root_temp = cp.deepcopy(root)
        node_list = []
        create_NL(root_temp)
        M = random.randint(2, K)
        for j in range(M):
            N = len(node_list)
            if N < 2: break
            P = random.randint(1, N - 1)
            root = prune_DT(root, P)
        accuracy_rtemp = validate_DT(root_temp, validation_set)
        accuracy_rbest = validate_DT(root_best, validation_set)
#         accuracy_rinit = validate_DT(root, validation_set)
#         print(accuracy_rtemp, accuracy_rbest, accuracy_rinit, "node", len(node_list))
        if accuracy_rtemp >= accuracy_rbest:
            root_best = cp.deepcopy(root_temp)
    print(validate_DT(root_best, validation_set))
    return root_best

def algorithm_2(L, K):
    global node_list
    root = None
    root = build_DT(root, training_set, attribute, -1, -1, "impurity")
    root_best = cp.deepcopy(root)
    print(validate_DT(root, validation_set))
    for i in range(L):
        root_temp = cp.deepcopy(root)
        node_list = []
        create_NL(root_temp)
        M = random.randint(2, K)
        for j in range(M):
            N = len(node_list)
            if N < 2: break
            P = random.randint(1, N - 1)
            root = prune_DT(root, P)
        accuracy_rtemp = validate_DT(root_temp, validation_set)
        accuracy_rbest = validate_DT(root_best, validation_set)
#         accuracy_rinit = validate_DT(root, validation_set)
#         print(accuracy_rtemp, accuracy_rbest, accuracy_rinit, "node", len(node_list))
        if accuracy_rtemp >= accuracy_rbest:
            root_best = cp.deepcopy(root_temp)
    print(validate_DT(root_best, validation_set))
    return root_best


# L = 100
# K = 10

root = algorithm_1(L, K)
root = algorithm_2(L, K)
if to_print: print_DT(root)
