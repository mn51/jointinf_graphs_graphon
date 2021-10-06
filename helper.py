import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import scipy as sp
from scipy import io
import multiprocess as mp
from numba import jit,njit,prange
from numba.typed import List
import numba as nb

@jit(nopython=True)
def soft_thresh(z, a):
    return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

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
def graphon_deg(x,graphon_fam='ER',p=None,p1=None,p0=None,gamma=None):
    if graphon_fam=='ER':
        return graphon(x,x)
    elif graphon_fam=='quad':
        if gamma==None:
            gamma = 1
        if p==None:
            p = 2
        return (gamma/p)*x**p + 1-gamma*(1-1/(p*(p+1)))
    elif graphon_fam=='SBM':
        if p1==None:
            p1=.75
        if p0==None:
            p0=1-p1
        if (type(x) is float):
            return (p1+p0)/2
        else:
            return (p1+p0)/2*np.ones(x.shape)
    else:
        print('Please choose ER, quad, or SBM graphon type.')
        return None

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
