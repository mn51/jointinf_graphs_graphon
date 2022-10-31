import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import scipy as sp
from scipy import io
from scipy import linalg
import multiprocess as mp
from numba import jit,njit,prange
from numba.typed import List
import numba as nb

from jointinf.spectemp import spectemp,spectemp_jgw1,spectemp_jgw2,wrapper_jgwi_probmat,wrapper_jgwi_nethist
from jointinf.pairdiff import pairdiff,pairdiff_jgw1,pairdiff_jgw2
from helper import graphon,soft_thresh,mykron

# ##############################################################################
# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Experiment: Vary RK, Z_K graph trials, Z_R signal trials
'''
#-------------------------------------------------------------------------------

#-------------------------
Z_K = 20
Z_R = 10
#-------------------------

#-------------------------
RK_RANGE = (np.logspace(2,5,7)).astype(int)
Z_RK_RANGE = len(RK_RANGE)
#-------------------------

X_SET = []
X_MRF_SET = []
X_smooth_SET = []

zeta_SET = []

S_TRUE = []
T_TRUE = []
W_TRUE = []

#-------------------------------------------------------------------------------
# Set parameters

#-------------------------
K  = 3
N  = (30*np.ones(K)).astype(int)
# RK = (1e4*np.ones(K)).astype(int)
L  = (N*(N-1)/2).astype(int)
G = np.sum(N)+10
J = int(G*(G-1)/2)
#-------------------------

#-------------------------
R = np.kron(np.ones((1,L[0])),np.eye(K))
Q = np.kron(np.eye(K), np.ones((1,L[0])))

# Indices of s for upper triangle, lower triangle, and diagonal
L_ind=[]
U_ind=[]
D_ind=[]
for k in range(K):
    L_ind.append(np.squeeze(np.where(np.matrix.flatten(np.tril(np.ones((N[k],N[k])))-np.eye(N[k]),'F'))))
    U_ind.append(np.squeeze(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k])))[1]*N[k] + np.squeeze(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k])))[0])
    D_ind.append(np.arange(N[k])*(N[k]+1))
L_ind_W=np.squeeze(np.where(np.matrix.flatten(np.tril(np.ones((G,G)))-np.eye(G),'F')))
#-------------------------

#-------------------------
graphon_fam = 'quad'
# p=.8
graphon_inds=(np.arange(G+1)/(G+1))[1:]
W=graphon(graphon_inds.reshape(-1,1),graphon_inds.reshape(1,-1),graphon_fam=graphon_fam)
W[np.eye(G)==1]=0
w=np.matrix.flatten(W,'F')[L_ind_W]

W_TRUE = w.copy()
#-------------------------

#-------------------------
Filt_Order = 3
h = np.random.normal(0,1,Filt_Order)
# h = np.random.random(F)

alpha = 5*np.random.random()
beta = np.random.random()
#-------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Z_K GRAPH TRIALS:
for z_K in range(Z_K):
    X_SET.append([])
    X_MRF_SET.append([])
    X_smooth_SET.append([])

    zeta_SET.append([])

    #-------------------------
    zeta = []
    zeta.append(np.sort(np.random.random(N[0])))
    T = []
    S = []
    T.append(graphon(zeta[0].reshape(-1,1),zeta[0].reshape(1,-1),graphon_fam=graphon_fam))
    S.append(np.zeros((N[0],N[0])))
    S[0][np.tril(np.ones((N[0],N[0])))-np.eye(N[0])==1] = np.random.binomial(1, T[0][np.tril(np.ones((N[0],N[0])))-np.eye(N[0]) == 1])
    S[0] += S[0].T
    for k in range(1,K):
        zeta.append(zeta[0])
        T.append(T[0])
        S.append(np.zeros((N[k],N[k])))
        S[k][np.tril(np.ones((N[k],N[k])))-np.eye(N[k])==1] = np.random.binomial(1, T[k][np.tril(np.ones((N[k],N[k])))-np.eye(N[k]) == 1])
        S[k] += S[k].T
    #-------------------------

    s = []
    t = []
    for k in range(K):
        s.append(np.matrix.flatten(S[k],'F'))
        s[k] = s[k][L_ind[k]].reshape(-1)
        t.append(np.matrix.flatten(T[k],'F'))
        t[k] = t[k][L_ind[k]].reshape(-1)
    S_TRUE.append(s)
    T_TRUE.append(t)

    z=[]
    for k in range(K):
        z.append(np.floor(zeta[k]*G).astype(int))
        z[k][z[k]>=G]=G-1

    I_z = []
    for k in range(K):
        zeta_idx=np.squeeze(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k])==1))
        I_z.append((np.eye(G**2)[(z[k][zeta_idx[0]])*G+z[k][zeta_idx[1]],:]+np.eye(G**2)[(z[k][zeta_idx[1]])*G+z[k][zeta_idx[0]],:])[:,L_ind_W])

    #-------------------------
    H = []
    H_MRF = []
    H_smooth = []
    for k in range(K):
        H.append(np.zeros((N[k],N[k])))
        H_MRF.append( sp.linalg.fractional_matrix_power( alpha*np.eye(N[k])+beta*S[k], -.5) )
        for f in range(Filt_Order):
            H[k] += h[f]*np.linalg.matrix_power(S[k],f)

        Lapl = np.diag(np.sum(S[k],axis=0)) - S[k]
        eval,evec=np.linalg.eigh(Lapl)
        h_eval = eval*(eval>0)
        h_eval[h_eval>0] = h_eval[h_eval>0]**(-.5)
        H_smooth.append(evec@np.diag(h_eval)@evec.T)

    while np.sum([np.sum(np.linalg.eigvals(H_MRF[k])<=0) for k in range(K)])>0:
        alpha = 5*np.random.random()
        beta = np.random.random()
        H_MRF = []
        for k in range(K):
            H_MRF.append( sp.linalg.fractional_matrix_power( alpha*np.eye(N[k])+beta*S[k], -.5) )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Z_R SIGNAL TRIALS:
    for z_R in range(Z_R):
        X_SET[z_K].append([])
        X_MRF_SET[z_K].append([])
        X_smooth_SET[z_K].append([])
        zeta_SET[z_K].append([])

        #-------------------------
        X_MAX = []
        X_MRF_MAX = []
        X_smooth_MAX = []
        for k in range(K):
            X_MAX.append(H[k]@np.random.normal(0,1,(N[k],RK_RANGE[-1])))
            X_MRF_MAX.append(H_MRF[k]@np.random.normal(0,1,(N[k],RK_RANGE[-1])))
            X_smooth_MAX.append(H_smooth[k]@np.random.normal(0,1,(N[k],RK_RANGE[-1])))
        #-------------------------

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # VARY RK IN RK_RANGE:
        for z_RK_RANGE in range(Z_RK_RANGE):
            RK = RK_RANGE[z_RK_RANGE]

            X = []
            X_MRF = []
            X_smooth = []
            for k in range(K):
                X.append(X_MAX[k][:,:RK])
                X_MRF.append(X_MRF_MAX[k][:,:RK])
                X_smooth.append(X_smooth_MAX[k][:,:RK])

            X_SET[z_K][z_R].append(X)
            X_MRF_SET[z_K][z_R].append(X_MRF)
            X_smooth_SET[z_K][z_R].append(X_smooth)
            zeta_SET[z_K][z_R].append(zeta)

X_DATASET = []
X_MRF_DATASET = []
X_smooth_DATASET = []

zeta_X_DATASET = []
zeta_X_MRF_DATASET = []
zeta_X_smooth_DATASET = []

for z_RK_RANGE in range(Z_RK_RANGE):
    X_DATASET.append([])
    X_MRF_DATASET.append([])
    X_smooth_DATASET.append([])

    zeta_X_DATASET.append([])
    zeta_X_MRF_DATASET.append([])
    zeta_X_smooth_DATASET.append([])

    for z_K in range(Z_K):
        X_DATASET[z_RK_RANGE].append([])
        X_MRF_DATASET[z_RK_RANGE].append([])
        X_smooth_DATASET[z_RK_RANGE].append([])

        zeta_X_DATASET[z_RK_RANGE].append([])
        zeta_X_MRF_DATASET[z_RK_RANGE].append([])
        zeta_X_smooth_DATASET[z_RK_RANGE].append([])

        for z_R in range(Z_R):
            X_DATASET[z_RK_RANGE][z_K].append(X_SET[z_K][z_R][z_RK_RANGE])
            X_MRF_DATASET[z_RK_RANGE][z_K].append(X_MRF_SET[z_K][z_R][z_RK_RANGE])
            X_smooth_DATASET[z_RK_RANGE][z_K].append(X_smooth_SET[z_K][z_R][z_RK_RANGE])

            zeta_X_DATASET[z_RK_RANGE][z_K].append((zeta_SET[z_K][z_R][z_RK_RANGE],X_SET[z_K][z_R][z_RK_RANGE]))
            zeta_X_MRF_DATASET[z_RK_RANGE][z_K].append((zeta_SET[z_K][z_R][z_RK_RANGE],X_MRF_SET[z_K][z_R][z_RK_RANGE]))
            zeta_X_smooth_DATASET[z_RK_RANGE][z_K].append((zeta_SET[z_K][z_R][z_RK_RANGE],X_smooth_SET[z_K][z_R][z_RK_RANGE]))

# ##############################################################################
# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Estimate graphs: Vary RK, Z_K graph trials, Z_R signal trials
'''
#-------------------------------------------------------------------------------
# Estimate graphs:

experiment_counter = 0

#-------------------------
print('ST:')
trial_counter = 0

S_ERROR_ST = np.zeros((Z_K,Z_R,Z_RK_RANGE))
S_ST = []

for z_RK_RANGE in range(Z_RK_RANGE):
    S_ST.append([])

    for z_K in range(Z_K):
        S_ST[z_RK_RANGE].append([])

        tic = time.perf_counter()
        pool = mp.Pool(5)
        results = pool.map_async(spectemp, X_DATASET[z_RK_RANGE][z_K])
        pool.close()
        pool.join()

        map_output = results.get()
        for z_R in range(Z_R):
            S_ST[z_RK_RANGE][z_K].append(map_output[z_R])

            S_ERROR_ST[z_K,z_R,z_RK_RANGE] = np.sum([np.linalg.norm(S_ST[z_RK_RANGE][z_K][z_R][k]-S_TRUE[z_K][k],1) for k in range(K)])/np.sum(L)
        toc = time.perf_counter()
        # print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.3f} s')
        if toc-tic<60:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.2f} s')
        elif toc-tic<3600:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/60:.2f} min')
        else:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/3600:.2f} hr')
        print(f'    S error: {np.mean(S_ERROR_ST[z_K,:,z_RK_RANGE]):.4f}')
        trial_counter += toc-tic

    print(f'    Mean S error: {np.mean(S_ERROR_ST[:,:,z_RK_RANGE]):.4f}')

if trial_counter<60:
    print(f'Total time: {trial_counter:.2f} s')
elif trial_counter<3600:
    print(f'Total time: {trial_counter/60:.2f} min')
else:
    print(f'Total time: {trial_counter/3600:.2f} hr')

experiment_counter+=trial_counter

print('')

##############################################################################
##############################################################################
#-------------------------
print('ST + JGW1:')
trial_counter = 0

S_ERROR_ST_JGW1 = np.zeros((Z_K,Z_R,Z_RK_RANGE))
S_ST_JGW1 = []

for z_RK_RANGE in range(Z_RK_RANGE):
    S_ST_JGW1.append([])

    for z_K in range(Z_K):
        S_ST_JGW1[z_RK_RANGE].append([])

        tic = time.perf_counter()
        pool = mp.Pool(5)
        results = pool.map_async(wrapper_jgwi_probmat, X_DATASET[z_RK_RANGE][z_K])
        pool.close()
        pool.join()

        map_output = results.get()
        for z_R in range(Z_R):
            S_ST_JGW1[z_RK_RANGE][z_K].append(map_output[z_R])

            S_ERROR_ST_JGW1[z_K,z_R,z_RK_RANGE] = np.sum([np.linalg.norm(S_ST_JGW1[z_RK_RANGE][z_K][z_R][k]-S_TRUE[z_K][k],1) for k in range(K)])/np.sum(L)
        toc = time.perf_counter()
        # print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.3f} s')
        if toc-tic<60:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.2f} s')
        elif toc-tic<3600:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/60:.2f} min')
        else:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/3600:.2f} hr')
        print(f'    S error: {np.mean(S_ERROR_ST_JGW1[z_K,:,z_RK_RANGE]):.4f}')
        trial_counter += toc-tic
    print(f'    Mean S error: {np.mean(S_ERROR_ST_JGW1[:,:,z_RK_RANGE]):.4f}')

if trial_counter<60:
    print(f'Total time: {trial_counter:.2f} s')
elif trial_counter<3600:
    print(f'Total time: {trial_counter/60:.2f} min')
else:
    print(f'Total time: {trial_counter/3600:.2f} hr')

experiment_counter+=trial_counter

print('')

# ##############################################################################
# ##############################################################################
#-------------------------
print('ST + JGW2:')
trial_counter = 0

S_ERROR_ST_JGW2 = np.zeros((Z_K,Z_R,Z_RK_RANGE))
W_ERROR_ST_JGW2 = np.zeros((Z_K,Z_R,Z_RK_RANGE))
S_ST_JGW2 = []
W_ST_JGW2 = []

for z_RK_RANGE in range(Z_RK_RANGE):
    S_ST_JGW2.append([])
    W_ST_JGW2.append([])

    for z_K in range(Z_K):
        S_ST_JGW2[z_RK_RANGE].append([])
        W_ST_JGW2[z_RK_RANGE].append([])

        tic = time.perf_counter()
        pool = mp.Pool(5)
        results = pool.map_async(wrapper_jgwi_nethist, zeta_X_DATASET[z_RK_RANGE][z_K])
        pool.close()
        pool.join()

        map_output = results.get()
        for z_R in range(Z_R):
            S_ST_JGW2[z_RK_RANGE][z_K].append(map_output[z_R][0])
            W_ST_JGW2[z_RK_RANGE][z_K].append(map_output[z_R][1])

            S_ERROR_ST_JGW2[z_K,z_R,z_RK_RANGE] = np.sum([np.linalg.norm(S_ST_JGW2[z_RK_RANGE][z_K][z_R][k]-S_TRUE[z_K][k],1) for k in range(K)])/np.sum(L)
            W_ERROR_ST_JGW2[z_K,z_R,z_RK_RANGE] = np.linalg.norm(W_ST_JGW2[z_RK_RANGE][z_K][z_R]-W_TRUE,1)/J
        toc = time.perf_counter()
        # print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.3f} s')
        if toc-tic<60:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.2f} s')
        elif toc-tic<3600:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/60:.2f} min')
        else:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/3600:.2f} hr')
        print(f'    S error: {np.mean(S_ERROR_ST_JGW2[z_K,:,z_RK_RANGE]):.4f}')
        print(f'    W error: {np.mean(W_ERROR_ST_JGW2[z_K,:,z_RK_RANGE]):.4f}')
        trial_counter += toc-tic

    print(f'    Mean S error: {np.mean(S_ERROR_ST_JGW2[:,:,z_RK_RANGE]):.4f}')
    print(f'    Mean W error: {np.mean(W_ERROR_ST_JGW2[:,:,z_RK_RANGE]):.4f}')


if trial_counter<60:
    print(f'Total time: {trial_counter:.2f} s')
elif trial_counter<3600:
    print(f'Total time: {trial_counter/60:.2f} min')
else:
    print(f'Total time: {trial_counter/3600:.2f} hr')

experiment_counter+=trial_counter

print('')

# ##############################################################################
# ##############################################################################
#-------------------------
print('PD:')
trial_counter = 0

S_ERROR_PD = np.zeros((Z_K,Z_R,Z_RK_RANGE))
S_PD = []

for z_RK_RANGE in range(Z_RK_RANGE):
    S_PD.append([])

    for z_K in range(Z_K):
        S_PD[z_RK_RANGE].append([])

        tic = time.perf_counter()
        pool = mp.Pool(5)
        results = pool.map_async(pairdiff, X_DATASET[z_RK_RANGE][z_K])
        pool.close()
        pool.join()

        map_output = results.get()
        for z_R in range(Z_R):
            S_PD[z_RK_RANGE][z_K].append(map_output[z_R])

            S_ERROR_PD[z_K,z_R,z_RK_RANGE] = np.sum([np.linalg.norm(S_PD[z_RK_RANGE][z_K][z_R][k]-S_TRUE[z_K][k],1) for k in range(K)])/np.sum(L)
        toc = time.perf_counter()
        # print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.3f} s')
        if toc-tic<60:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.2f} s')
        elif toc-tic<3600:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/60:.2f} min')
        else:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/3600:.2f} hr')
        print(f'    S error: {np.mean(S_ERROR_PD[z_K,:,z_RK_RANGE]):.4f}')
        trial_counter += toc-tic

    print(f'    Mean S error: {np.mean(S_ERROR_PD[:,:,z_RK_RANGE]):.4f}')

if trial_counter<60:
    print(f'Total time: {trial_counter:.2f} s')
elif trial_counter<3600:
    print(f'Total time: {trial_counter/60:.2f} min')
else:
    print(f'Total time: {trial_counter/3600:.2f} hr')

experiment_counter+=trial_counter

print('')

##############################################################################
##############################################################################
#-------------------------
print('PD + JGW1:')
trial_counter = 0

S_ERROR_PD_JGW1 = np.zeros((Z_K,Z_R,Z_RK_RANGE))
S_PD_JGW1 = []

for z_RK_RANGE in range(Z_RK_RANGE):
    S_PD_JGW1.append([])

    for z_K in range(Z_K):
        S_PD_JGW1[z_RK_RANGE].append([])

        tic = time.perf_counter()
        pool = mp.Pool(5)
        results = pool.map_async(pairdiff_jgw1, X_DATASET[z_RK_RANGE][z_K])
        pool.close()
        pool.join()

        map_output = results.get()
        for z_R in range(Z_R):
            S_PD_JGW1[z_RK_RANGE][z_K].append(map_output[z_R])

            S_ERROR_PD_JGW1[z_K,z_R,z_RK_RANGE] = np.sum([np.linalg.norm(S_PD_JGW1[z_RK_RANGE][z_K][z_R][k]-S_TRUE[z_K][k],1) for k in range(K)])/np.sum(L)
        toc = time.perf_counter()
        # print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.3f} s')
        if toc-tic<60:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.2f} s')
        elif toc-tic<3600:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/60:.2f} min')
        else:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/3600:.2f} hr')
        print(f'    S error: {np.mean(S_ERROR_PD_JGW1[z_K,:,z_RK_RANGE]):.4f}')
        trial_counter += toc-tic
    print(f'    Mean S error: {np.mean(S_ERROR_PD_JGW1[:,:,z_RK_RANGE]):.4f}')

if trial_counter<60:
    print(f'Total time: {trial_counter:.2f} s')
elif trial_counter<3600:
    print(f'Total time: {trial_counter/60:.2f} min')
else:
    print(f'Total time: {trial_counter/3600:.2f} hr')

experiment_counter+=trial_counter

print('')

# ##############################################################################
# ##############################################################################
#-------------------------
print('PD + JGW2:')
trial_counter = 0

S_ERROR_PD_JGW2 = np.zeros((Z_K,Z_R,Z_RK_RANGE))
W_ERROR_PD_JGW2 = np.zeros((Z_K,Z_R,Z_RK_RANGE))
S_PD_JGW2 = []
W_PD_JGW2 = []

for z_RK_RANGE in range(Z_RK_RANGE):
    S_PD_JGW2.append([])
    W_PD_JGW2.append([])

    for z_K in range(Z_K):
        S_PD_JGW2[z_RK_RANGE].append([])
        W_PD_JGW2[z_RK_RANGE].append([])

        tic = time.perf_counter()
        pool = mp.Pool(5)
        results = pool.map_async(pairdiff_jgw2, zeta_X_DATASET[z_RK_RANGE][z_K])
        pool.close()
        pool.join()

        map_output = results.get()
        for z_R in range(Z_R):
            S_PD_JGW2[z_RK_RANGE][z_K].append(map_output[z_R][0])
            W_PD_JGW2[z_RK_RANGE][z_K].append(map_output[z_R][1])

            S_ERROR_PD_JGW2[z_K,z_R,z_RK_RANGE] = np.sum([np.linalg.norm(S_PD_JGW2[z_RK_RANGE][z_K][z_R][k]-S_TRUE[z_K][k],1) for k in range(K)])/np.sum(L)
            W_ERROR_PD_JGW2[z_K,z_R,z_RK_RANGE] = np.linalg.norm(W_PD_JGW2[z_RK_RANGE][z_K][z_R]-W_TRUE,1)/J
        toc = time.perf_counter()
        # print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.3f} s')
        if toc-tic<60:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {toc-tic:.2f} s')
        elif toc-tic<3600:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/60:.2f} min')
        else:
            print(f'No. of signals {RK_RANGE[z_RK_RANGE]}, graph trial {z_K+1}: {(toc-tic)/3600:.2f} hr')
        print(f'    S error: {np.mean(S_ERROR_PD_JGW2[z_K,:,z_RK_RANGE]):.4f}')
        print(f'    W error: {np.mean(W_ERROR_PD_JGW2[z_K,:,z_RK_RANGE]):.4f}')
        trial_counter += toc-tic

    print(f'    Mean S error: {np.mean(S_ERROR_PD_JGW2[:,:,z_RK_RANGE]):.4f}')
    print(f'    Mean W error: {np.mean(W_ERROR_PD_JGW2[:,:,z_RK_RANGE]):.4f}')


if trial_counter<60:
    print(f'Total time: {trial_counter:.2f} s')
elif trial_counter<3600:
    print(f'Total time: {trial_counter/60:.2f} min')
else:
    print(f'Total time: {trial_counter/3600:.2f} hr')

experiment_counter+=trial_counter

print('')


if experiment_counter<60:
    print(f'Experiment time: {experiment_counter:.2f} s')
elif experiment_counter<3600:
    print(f'Experiment time: {experiment_counter/60:.2f} min')
else:
    print(f'Experiment time: {experiment_counter/3600:.2f} hr')

# ##############################################################################
# ##############################################################################
'''
Graph error: Vary RK, Z_K graph trials, Z_R signal trials
'''
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#-------------------------------------------------------------------------------
# Plot estimation error:

plt.figure()
plt.semilogx(RK_RANGE,np.mean(np.mean(S_ERROR_ST,axis=1),axis=0),'-o',c='blue')
plt.semilogx(RK_RANGE,np.mean(np.mean(S_ERROR_ST_JGW1,axis=1),axis=0),'--s',c='blue')
plt.semilogx(RK_RANGE,np.mean(np.mean(S_ERROR_ST_JGW2,axis=1),axis=0),':d',c='blue')

plt.semilogx(RK_RANGE,np.mean(np.mean(S_ERROR_PD,axis=1),axis=0),'-o',c='red')
plt.semilogx(RK_RANGE,np.mean(np.mean(S_ERROR_PD_JGW1,axis=1),axis=0),'--s',c='red')
plt.semilogx(RK_RANGE,np.mean(np.mean(S_ERROR_PD_JGW2,axis=1),axis=0),':d',c='red')

plt.xlabel('No. of signals')
plt.ylabel('Error')
plt.title('Graph error')
plt.grid(True)
plt.legend([\
            'Sparse','Sparse (Mod. 1)','Sparse (Mod. 2)',\
            'Pair. Diff.','Pair. Diff. (Mod. 1)','Pair. Diff. (Mod. 2)'\
            ])
