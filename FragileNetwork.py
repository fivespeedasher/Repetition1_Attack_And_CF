# -*- coding: utf-8 -*-
#TODO 测试其他的网络
#TODO 加入其他网络、画图、加入LD_attack
import networkx as nx
import matplotlib.pyplot as plt
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
        Attack m nodes which have the highest degree. Resetting L when attack next node. Return normalized CF
        m:  int
        Return->CFattack:   int
        """
        # Find the highest m nodes
        top_m_k = self.Find_HD(m)

        ## Processing degree 0
        self.L = np.where(self.L==0, self.L-1, self.L)
        Ltemp = np.copy(self.L)
        assert(id(Ltemp)!=id(self.L))

        assert(self.n == self.k.size)
        CF = np.zeros((self.n,1))
        # start to attack nodes in top_m_k
        for i in top_m_k:
            node = i
            count = 0
            while self.L[i] > 0:
                # delta_L regard each node broken can distribute load to others
                sum_Li = np.abs(self.A @ self.L) # for avoiding a negative Li become a positive one
                assert(sum_Li.any() != 0) 
                delta_L = (((self.L / sum_Li) @ self.L.T) * self.A)
                # delta_L = np.nan_to_num(delta_L)
                # self.A[i,:] = 0
                # self.A[:,i] = 0
                self.L[i] = -1 # for avoiding nan from division
                self.L = self.L+(delta_L[i,:].reshape(self.k.size,1))
                assert(self.L.size == self.k.size)

                new_break = np.argwhere(self.L>self.C)[:,0]
                if new_break.size ==0: break
                else:
                    i = new_break[0]
                    count += 1
                    new_break = new_break[1:]
            CF[node] = count
            self.L = np.copy(Ltemp)

        CF_normalized = np.sum(CF,axis=0) / (m * (self.n-1))
        return CF_normalized

def average_cf(times,nodes,T):
    """"
    times->int : means looping times and then we dividing initial_times to calculate average CF of the same T
    nodes, T -> int
    Return: CF_average -> int
    """
    init_times = times
    HD_CF_normalized_sum = 0
    while times:
        times -= 1
        G_BA = nx.barabasi_albert_graph(nodes,2)
        Adj = nx.to_numpy_array(G_BA)
        a_network = FragileNetwork(Adj,T)
        # attack 10 nodes
        HD_CF_normalized_sum += a_network.HD_attack(10) 

    HD_CF_average = HD_CF_normalized_sum / init_times
    return HD_CF_average

def visualize(t,HD_BA_CF,HD_WS_CF=[0],LD_BA_CF=[0],LD_WS_CF=[0]):
    """
    plot the CF_normalized after HD attack and LD attack
    Input ->  array
    Return -> None
    """
    fig = plt.figure()
    fig, (HD_BA) = plt.subplots(1,1, sharex=True)
    HD_BA.plot(t,HD_BA_CF)
    plt.show()
    return 0


if __name__ == '__main__':
    # Attack BA network, try 20 times
    T = np.arange(1,1.25,0.005)
    HD_CF = []
    for t in T:
        HD_CF.append(average_cf(20,200,t))
        print("t,HD_CF",t,HD_CF)
    visualize(T,HD_CF)
