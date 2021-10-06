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

from jointinf.helper import soft_thresh,graphon,mykron

# ##############################################################################
# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 2a: PD         pairdiff
################################################################################
@jit(nopython=True)
def in_wrapper_pairdiff(X):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph parameters:
    K =  len(X)
    N =  np.array([X[k].shape[0] for k in range(K)])
    RK = np.array([X[k].shape[1] for k in range(K)])
    L = np.array([int(N[k]*(N[k]-1)/2) for k in range(K)])

    L_ind_list=[]
    U_ind_list=[]
    D_ind_list=[]
    for k in range(K):
        L_ind_list.append(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[0]*N[k] + np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[1])
        U_ind_list.append(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[1]*N[k] + np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[0])
        D_ind_list.append(np.arange(N[k])*(N[k]+1))

    L_ind = np.zeros(np.sum(L))
    U_ind = np.zeros(np.sum(L))
    for k in range(K):
        L_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = L_ind_list[k] + np.sum(N[:k]**2)
        U_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = U_ind_list[k] + np.sum(N[:k]**2)

    D = np.zeros((int(K*(K-1)/2), K))
    upp_tri_ind = np.where(np.triu(np.ones((K, K)))-np.eye(K))
    if K > 2:
        for l in range(D.shape[0]):
            D[l, upp_tri_ind[0][l]] = 1
            D[l, upp_tri_ind[1][l]] = -1
    else:
        D[0,0]=1
        D[0,1]=-1
    W = np.kron(D, np.eye(L[0]))

    Q = np.zeros((K,np.sum(L)))
    for k in range(K):
        Q[k,np.sum(L[:k]):np.sum(L[:k])+L[k]]=1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    Sigma = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        Sigma[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = Sigma[:,int(L_ind[kl])]+Sigma[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    s_pd = np.random.binomial(1,.5,np.sum(L))*1.
    p_pd = s_pd.copy()
    l_pd = W@s_pd
    u1 = np.zeros(int(K*L[0]*(K-1)/2))
    u2 = np.zeros(np.sum(L))

    theta=10.
    rho1=10.
    rho2=10.
    eta=1.
    phi=10.

    eps_abs = 0
    eps_rel = 1e-3

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv(theta*M.T@M + rho1*W.T@W + rho2*np.eye(np.sum(L)))

    for admm_iter in range(int(1e3)):
        # s-update
        Phi_s2 = rho1*W.T@(l_pd-u1) + rho2*(p_pd-u2)
        s_pd = Phi_s1@Phi_s2

        # p-update
        p_last=p_pd.copy()
        ret = s_pd + u2
        p_pd = (ret-np.min(ret))/np.max((ret-np.min(ret)))
        p_pd[p_pd >= .5] = 1
        p_pd[p_pd <  .5] = 0

        # l-update
        l_last = l_pd.copy()
        l_pd = soft_thresh(W@s_pd + u1, eta/rho1)

        # u-update
        u1 = u1 + W@s_pd - l_pd
        u2 = u2 + s_pd - p_pd

        pr.append(
            np.linalg.norm(W@s_pd - l_pd,2) +
            np.linalg.norm(s_pd - p_pd,2)
            )
        dr.append(
            np.linalg.norm(rho1*W.T@(l_pd - l_last),2) +
            np.linalg.norm(rho2*(p_pd - p_last),2)
            )
        eps_pri.append(
            eps_abs*np.sqrt(len(l_pd)) + eps_rel*np.amax( np.array([np.linalg.norm(W@s_pd),np.linalg.norm(l_pd)]) ) +
            eps_abs*np.sqrt(len(p_pd)) + eps_rel*np.amax( np.array([(np.linalg.norm(s_pd)),np.linalg.norm(p_pd)]) )
            )
        eps_dua.append(
            eps_abs*np.sqrt(len(u1)) + eps_rel*(np.linalg.norm(rho1*W.T@u1)) +
            eps_abs*np.sqrt(len(u2)) + eps_rel*(np.linalg.norm(rho2*u2))
            )

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) and
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

    s_pd = []
    for k in range(K):
        s_pd.append(p_pd[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_pd

################################################################################
def pairdiff(X):
    K = len(X)

    X_NUMBA = List()
    [X_NUMBA.append(X[k]) for k in range(K)]

    s_pd = in_wrapper_pairdiff(X_NUMBA)
    for k in range(K):
        s_pd[k] = s_pd[k].astype(int)
    return s_pd





# ##############################################################################
# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 2b: PD+JGWI1   pairdiff_jgw1
################################################################################
@jit(nopython=True)
def in_wrapper_pairdiff_jgw1(X):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph parameters:
    K =  len(X)
    N =  np.array([X[k].shape[0] for k in range(K)])
    RK = np.array([X[k].shape[1] for k in range(K)])
    L = np.array([int(N[k]*(N[k]-1)/2) for k in range(K)])

    L_ind_list=[]
    U_ind_list=[]
    D_ind_list=[]
    for k in range(K):
        L_ind_list.append(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[0]*N[k] + np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[1])
        U_ind_list.append(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[1]*N[k] + np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k]))[0])
        D_ind_list.append(np.arange(N[k])*(N[k]+1))

    L_ind = np.zeros(np.sum(L))
    U_ind = np.zeros(np.sum(L))
    for k in range(K):
        L_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = L_ind_list[k] + np.sum(N[:k]**2)
        U_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = U_ind_list[k] + np.sum(N[:k]**2)

    D = np.zeros((int(K*(K-1)/2), K))
    upp_tri_ind = np.where(np.triu(np.ones((K, K)))-np.eye(K))
    if K > 2:
        for l in range(D.shape[0]):
            D[l, upp_tri_ind[0][l]] = 1
            D[l, upp_tri_ind[1][l]] = -1
    else:
        D[0,0]=1
        D[0,1]=-1
    W = np.kron(D, np.eye(L[0]))

    Q = np.zeros((K,np.sum(L)))
    for k in range(K):
        Q[k,np.sum(L[:k]):np.sum(L[:k])+L[k]]=1

    R = (1/K)*np.kron(np.ones((1,K)),np.eye(L[0]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    Sigma = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        Sigma[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = Sigma[:,int(L_ind[kl])]+Sigma[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    s_pd = np.random.binomial(1,.5,np.sum(L))*1.
    p_pd = s_pd.copy()
    l_pd = W@s_pd
    t_pd = R@s_pd
    u1 = np.zeros(int(K*L[0]*(K-1)/2))
    u2 = np.zeros(np.sum(L))
    u3 = np.zeros(L[0])

    theta=10.
    rho1=10.
    rho2=10.
    rho3=10.
    eta=1e-3
    phi=10.

    eps_abs = 0
    eps_rel = 1e-3

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv(theta*M.T@M + rho1*W.T@W + rho2*np.eye(np.sum(L)) + rho3*R.T@R)

    for admm_iter in range(int(1e3)):
        # s-update
        Phi_s2 = rho1*W.T@(l_pd-u1) + rho2*(p_pd-u2) + rho3*R.T@(t_pd-u3)
        s_pd = Phi_s1@Phi_s2

        # p-update
        p_last=p_pd.copy()
        ret = s_pd + u2
        p_pd = (ret-np.min(ret))/np.max((ret-np.min(ret)))
        p_pd[p_pd >= .5] = 1
        p_pd[p_pd <  .5] = 0

        # l-update
        l_last = l_pd.copy()
        l_pd = soft_thresh(W@s_pd + u1, eta/rho1)

        # t-update
        eps_t_thresh = 1e-6
        t_last = t_pd.copy()

        t_DCA = t_pd.copy()
        y_DCA = np.zeros(L[0])
        currently_updating = np.ones(L[0])==1

        for idx in range(int(1e2)):
            y_DCA[currently_updating] = 0
            y_DCA[currently_updating*(t_DCA>0)*(t_DCA<1)] = \
                K*np.log(t_DCA[currently_updating*(t_DCA>0)*(t_DCA<1)]/(1-t_DCA[currently_updating*(t_DCA>0)*(t_DCA<1)]))
            t_new = y_DCA/rho3 + R@s_pd + u3
            t_new[t_new>1]=1
            t_new[t_new<0]=0

            currently_updating[np.abs(t_new-t_DCA) <= eps_t_thresh] = False
            t_DCA = t_new.copy()

            if np.sum(currently_updating) <= 0:
                break
        t_pd = t_DCA.copy()

        # u-update
        u1 = u1 + W@s_pd - l_pd
        u2 = u2 + s_pd - p_pd
        u3 = u3 + R@s_pd - t_pd

        pr.append(
            np.linalg.norm(W@s_pd - l_pd,2) +
            np.linalg.norm(s_pd - p_pd,2) +
            np.linalg.norm(R@s_pd - t_pd,2)
            )
        dr.append(
            np.linalg.norm(rho1*W.T@(l_pd - l_last),2) +
            np.linalg.norm(rho2*(p_pd - p_last),2) +
            np.linalg.norm(rho3*R.T@(t_pd - t_last),2)
            )
        eps_pri.append(
            eps_abs*np.sqrt(len(l_pd)) + eps_rel*np.amax( np.array([np.linalg.norm(W@s_pd),np.linalg.norm(l_pd)]) ) +
            eps_abs*np.sqrt(len(p_pd)) + eps_rel*np.amax( np.array([np.linalg.norm(s_pd),np.linalg.norm(p_pd)]) ) +
            eps_abs*np.sqrt(len(t_pd)) + eps_rel*np.amax( np.array([np.linalg.norm(R@s_pd),np.linalg.norm(t_pd)]) )
            )
        eps_dua.append(
            eps_abs*np.sqrt(len(u1)) + eps_rel*(np.linalg.norm(rho1*W.T@u1)) +
            eps_abs*np.sqrt(len(u2)) + eps_rel*(np.linalg.norm(rho2*u2)) +
            eps_abs*np.sqrt(len(u3)) + eps_rel*(np.linalg.norm(rho3*R.T@u3))
            )

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) and
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

    s_pd = []
    for k in range(K):
        s_pd.append(p_pd[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_pd

################################################################################
def pairdiff_jgw1(X):
    K = len(X)

    X_NUMBA = List()
    [X_NUMBA.append(X[k]) for k in range(K)]

    s_pd = in_wrapper_pairdiff_jgw1(X_NUMBA)
    for k in range(K):
        s_pd[k] = s_pd[k].astype(int)
    return s_pd





# ##############################################################################
# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 2c: PD+JGWI2   pairdiff_jgw2
################################################################################
@jit(nopython=True)
def in_wrapper_pairdiff_jgw2(zeta,X):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph parameters:
    K =  len(X)
    N =  np.array([X[k].shape[0] for k in range(K)])
    RK = np.array([X[k].shape[1] for k in range(K)])
    L = np.array([int(N[k]*(N[k]-1)/2) for k in range(K)])

    G = int(np.sum(N)+10)
    J = int(G*(G-1)/2)

    low_tri_ind_list=[]
    L_ind_list=[]
    U_ind_list=[]
    D_ind_list=[]
    for k in range(K):
        low_tri_ind_list.append(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k])))
        L_ind_list.append(low_tri_ind_list[k][0]*N[k] + low_tri_ind_list[k][1])
        U_ind_list.append(low_tri_ind_list[k][1]*N[k] + low_tri_ind_list[k][0])
        D_ind_list.append(np.arange(N[k])*(N[k]+1))

    low_tri_ind_W = np.where(np.triu(np.ones((G,G)))-np.eye(G))
    L_ind_W = low_tri_ind_W[0]*G + low_tri_ind_W[1]
    U_ind_W = low_tri_ind_W[1]*G + low_tri_ind_W[0]

    L_ind = np.zeros(np.sum(L))
    U_ind = np.zeros(np.sum(L))
    for k in range(K):
        L_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = L_ind_list[k] + np.sum(N[:k]**2)
        U_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = U_ind_list[k] + np.sum(N[:k]**2)

    D = np.zeros((int(K*(K-1)/2), K))
    upp_tri_ind = np.where(np.triu(np.ones((K, K)))-np.eye(K))
    if K > 2:
        for l in range(D.shape[0]):
            D[l, upp_tri_ind[0][l]] = 1
            D[l, upp_tri_ind[1][l]] = -1
    else:
        D[0,0]=1
        D[0,1]=-1
    W = np.kron(D, np.eye(L[0]))

    # Differential matrix for graphon
    D1=np.concatenate((-np.eye(G-1),np.zeros((1,G-1))),axis=0)+np.concatenate((np.zeros((1,G-1)),np.eye(G-1)),axis=0)
    D2=np.concatenate((np.eye(G-2),np.zeros((2,G-2))),axis=0)+np.concatenate((np.zeros((1,G-2)),np.concatenate((-2*np.eye(G-2),np.zeros((1,G-2))),axis=0)),axis=0)+np.concatenate((np.zeros((2,G-2)),np.eye(G-2)),axis=0)

    d2 = np.kron(np.eye(G),D2.T)[:,L_ind_W]+np.kron(np.eye(G),D2.T)[:,U_ind_W]
    d1 = np.kron(D1.T,D1.T)[:,L_ind_W]+np.kron(D1.T,D1.T)[:,U_ind_W]

    # Network histogram matrix for graphs
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

    f=5
    F=[]
    for k in range(K):
        F.append(np.kron(np.eye(f),np.ones((int(np.ceil(N[k]/f)),int(np.ceil(N[k]/f))))-np.eye(int(np.ceil(N[k]/f))))[:int(N[k]),:int(N[k])])
        F[-1][np.sum(F[-1],axis=0)>0,:]/=(np.sum(F[-1],axis=0)[np.sum(F[-1],axis=0)>0]).reshape(-1,1)

    f_k = [(mykron(F[k],F[k])[:,L_ind_list[k]] + mykron(F[k],F[k])[:,U_ind_list[k]])[L_ind_list[k]] for k in range(K)]
    f = np.zeros((np.sum(L),np.sum(L)))
    for k in range(K):
        f[np.sum(L[:k]):np.sum(L[:k])+L[k],np.sum(L[:k]):np.sum(L[:k])+L[k]] = f_k[k]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graphon sampled points zeta

    z=[]
    for k in range(K):
        z.append(np.arange(int(N[k])))
        for z_idx in range(int(N[k])):
            z[k][z_idx] = int(zeta[k][z_idx]*G)
        z[k][z[k]>=G]=G-1

    I_z = [np.eye(G)[z[k],:] for k in range(K)]
    i_k = [(np.kron(I_z[k],I_z[k])[:,L_ind_W] + np.kron(I_z[k],I_z[k])[:,U_ind_W])[L_ind_list[k]] for k in range(K)]
    i_z = np.zeros((np.sum(L),J))
    for k in range(K):
        i_z[np.sum(L[:k]):np.sum(L[:k])+L[k],:] = i_k[k]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    Sigma = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        Sigma[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = Sigma[:,int(L_ind[kl])]+Sigma[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    s_pd = np.random.binomial(1,.5,np.sum(L))*1.
    p_pd = s_pd.copy()
    l_pd = W@s_pd
    w_pd = np.random.random(J)
    v_pd = w_pd.copy()
    t_pd = .5*(f@s_pd + i_z@w_pd)

    u1 = np.zeros(int(K*L[0]*(K-1)/2))
    u2 = np.zeros(np.sum(L))
    u3 = np.zeros(J)

    theta=10.
    rho1=10.
    rho2=10.
    rho3=10.
    eta=1e-3
    phi=10.
    lmbda=10.
    Lmbda=10.

    eps_abs = 0
    eps_rel = 1e-3

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv(theta*M.T@M + rho1*W.T@W + rho2*np.eye(np.sum(L)) + 2*Lmbda*f.T@f)
    Phi_w1 = np.linalg.inv( 4*lmbda*d2.T@d2 + 4*lmbda*d1.T@d1 + 2*Lmbda*i_z.T@i_z + rho3*np.eye(J) )

    for admm_iter in range(int(1e3)):
        # s-update
        gamma = np.zeros(np.sum(L))
        gamma[(t_pd>0)*(t_pd<1)] = np.log(t_pd[(t_pd>0)*(t_pd<1)]/(1-t_pd[(t_pd>0)*(t_pd<1)]))
        Phi_s2 = rho1*W.T@(l_pd-u1) + rho2*(p_pd-u2) + gamma + 2*Lmbda*f.T@(t_pd)
        s_pd = Phi_s1@Phi_s2

        # p-update
        p_last=p_pd.copy()
        ret = s_pd + u2
        p_pd = (ret-np.min(ret))/np.max((ret-np.min(ret)))
        p_pd[p_pd >= .5] = 1
        p_pd[p_pd <  .5] = 0

        # l-update
        l_last = l_pd.copy()
        l_pd = soft_thresh(W@s_pd + u1, eta/rho1)

        mu=.01
        pg_stop=1e-2
        t_hat = t_pd.copy()
        t_hat[t_hat>1.]=1.
        t_hat[t_hat<0.]=0.
        ind_t_hat=(t_hat>0.)*(t_hat<1.)
        Update_Indices = np.ones(np.sum(L))==1
        for pg_iter in range(int(2e2)):
            last_t=t_hat.copy()

            t_tild = last_t.copy()
            t_tild[ind_t_hat] = last_t[ind_t_hat] - mu*(s_pd[ind_t_hat]-last_t[ind_t_hat])/(last_t[ind_t_hat]*(last_t[ind_t_hat]-1))
            t_hat = (1/(4*Lmbda+1/mu))*( 2*Lmbda*(f@s_pd + i_z@w_pd) + (1/mu)*t_tild )
            t_hat[t_hat>1]=1
            t_hat[t_hat<0]=0
            ind_t_hat=(t_hat>0)*(t_hat<1)

            if np.linalg.norm(last_t-t_hat)**2<=pg_stop:
                break
        t_pd = t_hat.copy()

        # w-update
        Phi_w2 = Lmbda*i_z.T@t_pd + rho3*(v_pd - u3)
        w_pd = Phi_w1@Phi_w2

        # v-update
        v_last = v_pd.copy()
        v_pd = w_pd + u3
        v_pd[v_pd>1]=1
        v_pd[v_pd<0]=0

        # u-update
        u1 = u1 + W@s_pd - l_pd
        u2 = u2 + s_pd - p_pd
        u3 = u3 + w_pd - v_pd

        pr.append(
            np.linalg.norm(W@s_pd - l_pd,2) +
            np.linalg.norm(s_pd - p_pd,2) +
            np.linalg.norm(w_pd - v_pd,2)
            )
        dr.append(
            np.linalg.norm(rho1*W.T@(l_pd - l_last),2) +
            np.linalg.norm(rho2*(p_pd - p_last),2) +
            np.linalg.norm(rho3*(v_pd - v_last),2)
            )
        eps_pri.append(
            eps_abs*np.sqrt(len(l_pd)) + eps_rel*np.amax( np.array([np.linalg.norm(W@s_pd),np.linalg.norm(l_pd)]) ) +
            eps_abs*np.sqrt(len(p_pd)) + eps_rel*np.amax( np.array([np.linalg.norm(s_pd),np.linalg.norm(p_pd)]) ) +
            eps_abs*np.sqrt(len(w_pd)) + eps_rel*np.amax( np.array([np.linalg.norm(w_pd),np.linalg.norm(v_pd)]) )
            )
        eps_dua.append(
            eps_abs*np.sqrt(len(u1)) + eps_rel*(np.linalg.norm(rho1*W.T@u1)) +
            eps_abs*np.sqrt(len(u2)) + eps_rel*(np.linalg.norm(rho2*u2)) +
            eps_abs*np.sqrt(len(u3)) + eps_rel*(np.linalg.norm(rho3*u3))
            )

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) and
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

    s_pd = []
    for k in range(K):
        s_pd.append(p_pd[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_pd, v_pd

################################################################################
def pairdiff_jgw2(zeta_X):
    K = len(zeta_X[0])

    zeta_NUMBA = List()
    [zeta_NUMBA.append(zeta_X[0][k]) for k in range(K)]

    X_NUMBA = List()
    [X_NUMBA.append(zeta_X[1][k]) for k in range(K)]

    s_pd,w_pd = in_wrapper_pairdiff_jgw2(zeta_NUMBA,X_NUMBA)
    for k in range(K):
        s_pd[k] = s_pd[k].astype(int)
    return s_pd,w_pd
