# -*- coding: utf-8 -*-
#TODO 测试其他的网络
#TODO 加入其他网络、画图、加入LD_attack
import networkx as nx
from networkx.classes.function import degree
from networkx.convert_matrix import to_numpy_matrix
from networkx.generators.classic import complete_graph
import numpy as np
from numpy.core.fromnumeric import shape, size

class FragileNetwork(object):
    def __init__(self,A,t) -> None:
        """
        Input: A: Adjacency matrix
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
        assert(type(A) is np.ndarray)
        self.n = A.shape[0]
        self.t = t
        # self.A = np.array(nx.to_numpy_matrix(self.G) )
        self.A = A
        self.k = np.sum(self.A,axis=0).reshape(self.n,1)[:,0] # k: m0*1 array
        self.L = 1*self.k.reshape(self.n,1) # beta = 1
        self.C = self.t*self.L
    def Find_HD(self,m) -> np.ndarray:
        """
        Find the highest m nodes, return nodes' index, 
        soting by descending and choosing last same biggest nodes ramdonly.
        m:  int
        Return->k_HD:   1*m array
        """
        # self.k = np.array([2,5,6,8,9,10,2,5,8,5,8]).reshape(1,11) # for testing the randomness
        assert(self.C.size == self.k.size)
        s_k = np.sort(self.k,axis=0)[::-1]  # axis=1 means using the colum label to sort, bigger first
        s_k_loc = np.argsort(self.k, axis=0)[::-1]  # index of sorting list
        assert(self.k[s_k_loc[0]] == s_k[0])
        top_m_loc = s_k_loc[0:m] # stores indexs of biggest nodes
        top_m_k = s_k[0:m] # nodes before random
        assert(top_m_k.ndim==1)
        print("Biggest nodes by sorting without random:",top_m_loc)

        # using randomness
        last_bigger_number = int(top_m_k[m-1])
        last_bigger_number_total = np.argwhere(self.k == last_bigger_number).flatten() # all matched indices about the last bigger number
        last_bigger_number_need = np.argwhere(top_m_k == last_bigger_number).flatten()
        needs = last_bigger_number_need.size
        assert(last_bigger_number_total.size>=needs)
        last_bigger_number_need = np.random.choice(last_bigger_number_total,needs,replace=False) # random choose indices with no repeat
        # assert(last_bigger_number_need.ndim == 1)
        top_m_loc = np.concatenate((top_m_loc[0:-needs],last_bigger_number_need))
        assert(top_m_loc.size == m)
        print("Attack nodes:", top_m_loc)
        # top_m_k = self.k[top_m_loc] # final random biggest number
        return top_m_loc.reshape(m,1)

    def HD_attack(self,m) -> int:
        """
        Attack m nodes which have the highest degree, return CFattack
        m:  int
        Return->CFattack:   int
        """
        # Find the highest m nodes
        top_m_k = self.Find_HD(m)

        ## Processing degree 0
        self.L = np.where(self.L==0, self.L-1, self.L)
        Ltemp = self.L

        assert(self.n == self.k.size)
        CF = np.zeros((self.n,1))
        # start to attack nodes in top_m_k
        for i in top_m_k:
            node = i
            while self.L[i] > 0:

                # delta_L regard each node broken can distribute load to others
                sum_Li = np.abs(self.A @ self.L) # for avoiding a negative Li become a positive one 
                delta_L = (((self.L / sum_Li) @ self.L.T) * self.A)
                delta_L = np.nan_to_num(delta_L)
                # self.A[i,:] = 0
                # self.A[:,i] = 0
                self.L[i] = -1 # for avoiding nan from division
                self.L = self.L+(delta_L[i,:].reshape(self.k.size,1))
                assert(self.L.size == self.k.size)

                new_broken = np.argwhere(self.L>self.C)[:,0]
                if new_broken.size ==0: break
                else:
                    i = new_broken[0]
                    new_broken = new_broken[1:]
            CF[node] = np.argwhere(self.L[:,0] < 0).size-1
            self.L = Ltemp

        CF_normalized = np.sum(CF,axis=0) / (m * (self.n-1))
        print(CF_normalized)
        
if __name__ == '__main__':
    G_BA = nx.barabasi_albert_graph(200,10)
    Adj = nx.to_numpy_array(G_BA)
    c_network = FragileNetwork(Adj,1.1)
    attack_ind = c_network.HD_attack(3)