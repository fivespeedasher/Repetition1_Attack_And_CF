# -*- coding: utf-8 -*-

import networkx as nx
from networkx.classes.function import degree
from networkx.convert_matrix import to_numpy_matrix
from networkx.generators.classic import complete_graph
import numpy as np
from numpy.core.fromnumeric import shape, size

class FragileNetwork(object):
    def __init__(self,m0,t) -> None:
        """
        m0: int
            Given nodes whice consist a complete network
        t:  int
            load capacity's coefficient

        n:  int
            nodes number
        k:  1*n array
            nodes degree
        L:  1*n array
            initial load
        C:  1*n array
            capacity of load
        """
        self.G = nx.complete_graph(m0)
        self.n = m0
        self.t = t
        self.A = np.array(nx.to_numpy_matrix(self.G) )
        self.k = np.sum(self.A,axis=0) # k: 1*m0 array
        self.L = 1*self.k # beta = 1
        self.C = self.t*self.L
        self.CF = np.zeros((1,m0))
    def Find_HD(self,m) -> np.ndarray:
        """
        Attack the highest m nodes, retrun each the number of break nodes caused by the node which broken before.
        m:  int
        Return->k_HD:   1*m array
        """
        # self.k = np.array([2,5,6,8,9,10,2,5,8,5,8]) # for testing the randomness
        s_k = np.sort(self.k,axis=0)[::-1]  # axis=1 means using the colum label to sort, bigger first
        s_k_loc = np.argsort(self.k, axis=0)[::-1]  # index of sorting list
        assert(self.k[s_k_loc[0]] == s_k[0])
        top_m_loc = s_k_loc[0:m] # TODO 随机切片
        top_m_k = s_k[0:m] # nodes before random
        print(top_m_loc)

        # using randomness
        last_bigger_number = top_m_k[m-1]
        last_bigger_number_total = np.argwhere(self.k == last_bigger_number).flatten() # all matched indices about the last bigger number
        last_bigger_number_need = np.argwhere(top_m_k == last_bigger_number).flatten()
        needs = last_bigger_number_need.size
        assert(last_bigger_number_total.size>needs)
        last_bigger_number_need = np.random.choice(last_bigger_number_total,needs,replace=False) # random choose indices with no repeat
        assert(last_bigger_number_need.ndim == 1)
        top_m_loc = np.concatenate((top_m_loc[0:-needs],last_bigger_number_need))
        assert(top_m_loc.size == m)
        print("Attack nodes:", top_m_loc)
        # top_m_k = self.k[top_m_loc] # final random biggest number
        return top_m_k

    def HD_attack(self,m) -> int:
        """
        Attack m nodes which have the highest degree, return CFattack
        m:  int
        Return->CFattack:   int
        """


if __name__ == '__main__':
    c_network = FragileNetwork(5,1.2)
    attack_ind = c_network.Find_HD(3) # the nodes (indices of self.k) where we attack