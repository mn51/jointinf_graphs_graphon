import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy as sp
from scipy import io
from scipy import linalg
from scipy import stats

from numba import jit,njit,prange
from numba.typed import List
import numba as nb

import networkx as nx
import time
import timeit
# import cvxpy as cp

#---------------------------------------------------------------
# Symmetric matrix to lower triangle vector
def mat2lowtri(A):
    (N,N) = A.shape
    L = int(N*(N-1)/2)

    low_tri_indices = np.where(np.triu(np.ones((N,N)))-np.eye(N))
    a = A[low_tri_indices[1],low_tri_indices[0]]
    return a

#---------------------------------------------------------------
# Lower triangle vector to symmetric matrix
def lowtri2mat(a):
    L = len(a)
    N = int(.5 + np.sqrt(2*L + .25))

    A = np.full((N,N),0,dtype=type(a[0]))
    low_tri_indices = np.where(np.triu(np.ones((N,N)))-np.eye(N))
    A[low_tri_indices[1],low_tri_indices[0]] = a
    A += A.T
    return A

#---------------------------------------------------------------
# Soft thresholding for l1-norm proximal operator
@njit
def soft_thresh(z,lmbda):
    return (z+lmbda*(-2*(z>0)+1))*(np.abs(z)>=lmbda)

#---------------------------------------------------------------
# Numba-friendly Kronecker product for matrices A and B
@jit(nopython=True)
def mykron(A,B):
    (ar,ac) = A.shape
    (br,bc) = B.shape
    C = np.zeros((ar*br,ac*bc))
    for i in range(ar):
        rjmp = i*br
        for k in range(br):
            irjmp = rjmp + k
            slice = B[k,:]
            for j in range(ac):
                cjmp = j*bc
                C[irjmp,cjmp:cjmp+bc] = A[i,j]*slice
    return C

# ------------------------------------------------------------------------------
# Sample graph from graphon
# def graph_from_graphon(N,theta=None,nKnots=None,graphon_type='ER'):
def graph_from_graphon(N,graphon_type='ER'):
    if graphon_type=='ER':
        edge_prob = .2
        G = np.array(nx.to_numpy_matrix(nx.generators.random_graphs.fast_gnp_random_graph(n=N,p=edge_prob)))
        G = G*np.tril((1-np.eye(N)))
        G += G.T
        return G
    elif graphon_type=='SBM':
        P = .9
        Q = .2
        SBM_mat = P*np.eye(2) + Q*(1-np.eye(2))
        # node_labels = np.concatenate((np.zeros(int(N/2)),np.ones(int(N/2)))).astype(int)
        node_labels = np.sort(np.random.binomial(1,.5,N))
        probmat = SBM_mat[node_labels.reshape(1,-1),node_labels.reshape(-1,1)]
        G = np.random.binomial(1,probmat)
        G = G*np.tril((1-np.eye(N)))
        G += G.T
        return G
    elif graphon_type=='quad':
        zeta = np.sort(np.random.random(N))
        Prob_mat = .5*(zeta.reshape(-1,1)**2+zeta.reshape(1,-1)**2)
        G = np.random.binomial(1,Prob_mat)
        G = G*np.tril((1-np.eye(N)))
        G += G.T
        return G
    elif graphon_type=='grid':
        k_grid=6
        G = np.array(nx.to_numpy_matrix(nx.generators.random_graphs.watts_strogatz_graph(n=N,k=k_grid,p=0)))
        G = G*np.tril((1-np.eye(N)))
        G += G.T
        return G
    elif graphon_type=='similarity':
        zeta = np.sort(np.random.random(N))
        Prob_mat = .8*np.exp(-2*np.abs(zeta.reshape(-1,1)-zeta.reshape(1,-1)))
        G = np.random.binomial(1,Prob_mat)
        G = G*np.tril((1-np.eye(N)))
        G += G.T
        return G
    # elif graphon_type=='theta':
    #     if theta==None:
    #         print('No theta given.')
    #     if nKnots==None and tau==None:
    #         print('Not enough information given.')
    #     elif tau==None:
    #         tau = np.concatenate(([0],np.linspace(0,1,nKnots),[1]))
    #     W_samp = byBSpline(tau=tau,P_mat=None,theta=theta_cc_centroids[0][k],order=1)
    #     G = GraphByGraphon(graphon=W_centroids[k],Us_real=zeta,N=N_samp[k][i])
    #     G = G.A
    else:
        print('Improper graphon type given.')

# ------------------------------------------------------------------------------
# Return graphon at given points
def graphon(x,y,graphon_fam='ER',p=None,p1=None,p0=None,gamma=None):
    if graphon_fam=='ER':
        if p==None:
            p = .5
        if (type(x) is float) and (type(y) is float):
            return p
        elif (type(x) is not float) and (type(y) is not float):
            return p*np.ones((x.shape[0]*y.shape[0],x.shape[1]*y.shape[1]))
        elif (type(x) is float) and (type(y) is not float):
            return p*np.ones(len(y))
        elif (type(x) is not float) and (type(y) is float):
            return p*np.ones(len(x))
    elif graphon_fam=='quad':
        if gamma==None:
            gamma = 1
        if p==None:
            p = 2
        return gamma*(x.reshape(-1,1)**p + y.reshape(1,-1)**p)/p + (1-gamma)
    elif graphon_fam=='SBM':
        if p1==None:
            p1=.75
        if p0==None:
            p0=1-p1
        return p1*(((x>.5)*(y>.5))+((x<=.5)*(y<=.5))) + \
               p0*(((x<=.5)*(y>.5))+((x>.5)*(y<=.5)))
    else:
        print('Please choose ER, quad, or SBM graphon type.')
        return None
