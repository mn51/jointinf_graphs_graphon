import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import scipy as sp
from scipy import io
from numba import jit,njit,prange
from numba.typed import List
import numba as nb

from helper import *

# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# sparse_stat
@jit(nopython=True)
def in_wrapper_sparse_stat(X,
                           theta=10, rho1=10, rho2=10, phi=10, eta=1e-1,
                           eps_abs=0.,eps_rel=1e-3,
                           num_admm_iters=1000,num_subadmm_iters=200
                           ):
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    M_FULL = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        M_FULL[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Q = np.zeros((K,np.sum(L)))
    for k in range(K):
        Q[k,np.sum(L[:k]):np.sum(L[:k])+L[k]]=1

    s_st = np.random.binomial(1,.5,np.sum(L))*1.
    p_st = s_st.copy()
    q_st = Q@s_st-1
    u1 = np.zeros(np.sum(L))
    u2 = np.zeros(K)

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    Phi_s1 = np.linalg.inv(theta*M.T@M + (rho1+phi)*np.eye(np.sum(L)) + rho2*Q.T@Q)

    for admm_iter in range(num_admm_iters):
        # s-update
        s1 = s_st.copy()
        s2 = s_st.copy()
        w = np.zeros(np.sum(L))

        pr2 = []
        dr2 = []
        eps_pri2 = []
        eps_dua2 = []
        for admm_iter2 in range(num_subadmm_iters):
            s1 = soft_thresh(s2-w,2*eta/phi)
            Phi_s2 = rho1*(p_st-u1) + rho2*Q.T@(q_st+1-u2) + phi*(s1+w)
            s2_last = s2.copy()
            s2 = Phi_s1@Phi_s2
            w = w + s1 - s2

            pr2.append(np.linalg.norm(s1-s2))
            dr2.append(np.linalg.norm(phi*(s2-s2_last)))
            eps_pri2.append( eps_abs*len(s1) + eps_rel*np.amax(np.array([np.linalg.norm(s1),np.linalg.norm(s2)])) )
            eps_dua2.append( eps_abs*len(s1) + eps_rel*np.linalg.norm(phi*w) )
            if (
                (np.sum(np.array(pr2)<=np.array(eps_pri2))>0) and
                (np.sum(np.array(dr2)<=np.array(eps_dua2))>0)
                ):
                break

        s_st = s2.copy()

        # p-update
        p_last=p_st.copy()
        ret = s_st + u1
        p_st = (ret-np.min(ret))/np.max((ret-np.min(ret)))
        p_st[p_st >= .5] = 1
        p_st[p_st <  .5] = 0

        # q-update
        q_last=q_st.copy()
        q_st = Q@s_st - 1 + u2
        q_st[q_st<0]=0

        # u-update
        u1 = u1 + s_st - p_st
        u2 = u2 + Q@s_st - q_st - 1

        pr.append(
            np.linalg.norm(s_st-p_st,2) +
            np.linalg.norm(Q@s_st-q_st-1,2)
            )
        dr.append(
            np.linalg.norm(rho1*(p_st-p_last),2) +
            np.linalg.norm(rho2*Q.T@(q_st-q_last),2)
            )
        eps_pri.append(
            eps_abs*np.sqrt(len(s_st)) + eps_rel*np.amax( np.array([np.linalg.norm(s_st),np.linalg.norm(p_st)]) ) +
            eps_abs*np.sqrt(len(s_st)) + eps_rel*np.amax( np.array([(np.linalg.norm(Q@s_st)),np.linalg.norm(q_st),np.linalg.norm(np.ones(K))]) )
            )
        eps_dua.append(
            eps_abs*np.sqrt(len(u1)) + eps_rel*(np.linalg.norm(rho1*u1)) +
            eps_abs*np.sqrt(len(u2)) + eps_rel*(np.linalg.norm(rho2*Q.T@u2))
            )

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) and
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

    s_est = []
    for k in range(K):
        s_est.append(p_st[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_est

def sparse_stat(X,
                theta=10, rho1=10, rho2=10, phi=10, eta=1e-1,
                eps_abs=0.,eps_rel=1e-3,
                num_admm_iters=1000,num_subadmm_iters=200
                ):
    K = len(X)

    X_NUMBA = List()
    [X_NUMBA.append(X[k]) for k in range(K)]

    s_est = in_wrapper_sparse_stat(X_NUMBA,
                                   theta=theta, rho1=rho1, rho2=rho2, phi=phi, eta=eta,
                                   eps_abs=eps_abs, eps_rel=eps_rel,
                                   num_admm_iters=num_admm_iters,num_subadmm_iters=num_subadmm_iters
                                   )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est




























# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# stat_mod1
@jit(nopython=True)
def in_wrapper_stat_mod1(X,
                                 rho1=10,rho2=10,theta=10,
                                 eps_abs=0.,eps_rel=1e-3,
                                 eps_t_thresh = 1e-6,
                                 num_admm_iters=1000,num_dca_iters=200
                                 ):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph parameters:
    K =  len(X)
    N =  X[0].shape[0]
    RK = X[0].shape[1]
    L = int(N*(N-1)/2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Manipulating parameters:

    # Indices of s for upper triangle, lower triangle, and diagonal
    U_ind = np.where(np.triu(np.ones((N, N)))-np.eye(N))[1]*N + np.where(np.triu(np.ones((N, N)))-np.eye(N))[0]
    U_ind = (np.kron(np.ones((K)), U_ind).reshape(-1, 1) +
             np.kron((np.arange(K).reshape(-1, 1)*(N**2)), np.ones((L, 1)))).reshape(-1)

    L_ind = np.where(np.triu(np.ones((N, N)))-np.eye(N))[0]*N + np.where(np.triu(np.ones((N, N)))-np.eye(N))[1]
    L_ind = (np.kron(np.ones((K)), L_ind).reshape(-1, 1) +
             np.kron((np.arange(K).reshape(-1, 1)*(N**2)), np.ones((L, 1)))).reshape(-1)

    D_ind = np.arange(N)*(N+1)
    D_ind = (np.kron(np.ones((K)), D_ind).reshape(-1, 1) +
             np.kron((np.arange(K).reshape(-1, 1)*(N**2)), np.ones((N, 1)))).reshape(-1)

    # Q for counting edges
    Q = np.kron(np.eye(K), np.ones((1, L)))

    # R for calculating average graph (probability matrix)
    R = (1/K)*np.kron(np.ones((1,K)),np.eye(L))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Estimated covariances
    C = []
    for k in range(K):
        C.append((X[k]@X[k].T)/RK)
    M_FULL = np.zeros((K*N**2, K*N**2))
    for k in range(K):
        M_FULL += np.kron(np.diag(np.eye(K)[k, :]), -np.kron(C[k],
                         np.eye(N)) + np.kron(np.eye(N), C[k]))
    # M for vectorized covariance constraint
    M = np.zeros((K*N**2,K*L))
    for kl in range(K*L):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Initialize

    pr1 = []
    pr2 = []
    pr = []

    dr1 = []
    dr2 = []
    dr = []

    eps_pri1=[]
    eps_pri2=[]
    eps_pri=[]

    eps_dua1=[]
    eps_dua2=[]
    eps_dua=[]

    # -------------------
    # s-initialize
    s_est = np.random.binomial(1,.5,K*L)*1.

    # -------------------
    # p-initialize
    p_est = s_est
    for k in range(K):
        p_est[k*L:(k+1)*L] = (s_est[k*L:(k+1)*L] - np.min(s_est[k*L:(k+1)*L])) / \
            np.max((s_est[k*L:(k+1)*L] - np.min(s_est[k*L:(k+1)*L])))
    p_est[p_est > .5]  = 1
    p_est[p_est <= .5] = 0

    # -------------------
    # t-initialize
    t_est = R@s_est

    # -------------------
    # u-initialize
    u1 = np.zeros(L)
    u2 = np.zeros(K*L)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Choose parameters

    Phi_s1 = np.linalg.inv(theta*M.T@M + rho1*R.T@R + rho2*np.eye(K*L))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Estimation
    for admm_iter in range(num_admm_iters):
        # ----------------
        # s-update
        Phi_s2 = rho1*R.T@(t_est-u1) + rho2*(p_est-u2)
        s_est = Phi_s1@Phi_s2

        # ----------------
        # p-update
        p_last=p_est.copy()
        ret = s_est+u2
        for k in range(K):
            p_est[k*L:(k+1)*L] = (ret[k*L:(k+1)*L] - np.min(ret[k*L:(k+1)*L])) / \
                np.max((ret[k*L:(k+1)*L] - np.min(ret[k*L:(k+1)*L])))
        p_est[p_est > .5] = 1
        p_est[p_est <= .5] = 0

        # ----------------
        # t-update
        t_last=t_est.copy()

        if rho1 == 0:
            t_DCA = R@s_est + u1
            t_DCA[t_DCA > 1] = 1
            t_DCA[t_DCA < 0] = 0
        else:
            t_DCA = t_est.copy()
            y_DCA = np.zeros(L)
            Update_Indices = np.ones(L) == 1

            for idx in range(num_dca_iters):
                y_DCA[Update_Indices] = 0
                y_DCA[Update_Indices*(t_DCA > 0)*(t_DCA < 1)] = \
                    K*np.log(t_DCA[Update_Indices*(t_DCA > 0)*(t_DCA < 1)] /
                             (1-t_DCA[Update_Indices*(t_DCA > 0)*(t_DCA < 1)]))

                t_update = y_DCA/rho1 + R@s_est + u1
                t_update[t_update > 1] = 1
                t_update[t_update < 0] = 0

                Update_Indices[np.abs(t_update-t_DCA) <= eps_t_thresh] = False
                t_DCA = t_update.copy()

                if np.sum(Update_Indices) <= 0:
                    break
        t_est = t_DCA.copy()

        # ----------------
        # u-updates
        u1 = u1 + R@s_est - t_est
        u2 = u2 + s_est - p_est

        # ----------------

        pr1.append(np.linalg.norm(R@s_est - t_est))
        pr2.append(np.linalg.norm(s_est - p_est))
        pr.append(pr1[-1]+pr2[-1])

        dr1.append(np.linalg.norm(rho1*R.T@(t_est-t_last)))
        dr2.append(np.linalg.norm(rho2*(p_est-p_last)))
        dr.append(dr1[-1]+dr2[-1])

        eps_pri1.append( eps_abs*np.sqrt(len(s_est)) + eps_rel*np.amax(np.array([np.linalg.norm(R@s_est),np.linalg.norm(t_est)])) )
        eps_pri2.append( eps_abs*np.sqrt(len(s_est)) + eps_rel*np.amax(np.array([np.linalg.norm(s_est),np.linalg.norm(p_est)])) )
        eps_pri.append(eps_pri1[-1]+eps_pri2[-1])

        eps_dua1.append( eps_abs*np.sqrt(len(u1)) + eps_rel*np.linalg.norm(rho1*R.T@u1) )
        eps_dua2.append( eps_abs*np.sqrt(len(u2)) + eps_rel*np.linalg.norm(rho2*u2) )
        eps_dua.append(eps_dua1[-1]+eps_dua2[-1])

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) or
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

        # End estimation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    s_ret = []
    t_ret = []
    for k in range(K):
        s_ret.append(p_est[k*L:(k+1)*L])
        t_ret.append(t_est)

    return s_ret, t_ret

def stat_mod1(X,
              rho1=10,rho2=10,theta=10,
              eps_abs=0.,eps_rel=1e-3,
              eps_t_thresh = 1e-6,
              num_admm_iters=1000,num_dca_iters=200
              ):
    K = len(X)

    X_NUMBA = List()
    [X_NUMBA.append(X[k]) for k in range(K)]

    s_est, t_est = in_wrapper_stat_mod1(X_NUMBA,
                                        rho1=rho1,rho2=rho2,theta=theta,
                                        eps_abs=eps_abs, eps_rel=eps_rel,
                                        eps_t_thresh=eps_t_thresh,
                                        num_admm_iters=num_admm_iters,num_dca_iters=num_dca_iters
                                        )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est, t_est




























# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# stat_mod2
@jit(nopython=True)
def in_wrapper_stat_mod2(zeta,X,
                         theta=10,rho1=10,rho2=10,phi=10,eta=1e-1,lmbda=1,Lmbda=10,
                         eps_abs=0.,eps_rel=1e-3,
                         num_bins=5,pg_step_size=.01,pg_stop=1e-2,
                         num_admm_iters=1000,num_pg_iters=200,
                         G=None
                         ):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph parameters:
    K =  len(X)
    N =  np.array([X[k].shape[0] for k in range(K)])
    RK = np.array([X[k].shape[1] for k in range(K)])
    L = np.array([int(N[k]*(N[k]-1)/2) for k in range(K)])

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
    D_ind_W = np.arange(G)*(G+1)

    L_ind = np.zeros(np.sum(L))
    U_ind = np.zeros(np.sum(L))
    for k in range(K):
        L_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = L_ind_list[k] + np.sum(N[:k]**2)
        U_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = U_ind_list[k] + np.sum(N[:k]**2)

    # Differential matrix for graphon
    D1=np.concatenate((-np.eye(G-1),np.zeros((1,G-1))),axis=0)+np.concatenate((np.zeros((1,G-1)),np.eye(G-1)),axis=0)
    D2=np.concatenate((np.eye(G-2),np.zeros((2,G-2))),axis=0)+np.concatenate((np.zeros((1,G-2)),np.concatenate((-2*np.eye(G-2),np.zeros((1,G-2))),axis=0)),axis=0)+np.concatenate((np.zeros((2,G-2)),np.eye(G-2)),axis=0)

    d2_D = np.kron(np.eye(G),D2.T)[:,D_ind_W]
    d2_L = np.kron(np.eye(G),D2.T)[:,L_ind_W]+np.kron(np.eye(G),D2.T)[:,U_ind_W]
    d2 = np.zeros((d2_D.shape[0],G+J))
    d2[:,:G] = d2_D
    d2[:,G:] = d2_L
    d1_D = np.kron(D1.T,D1.T)[:,D_ind_W]
    d1_L = np.kron(D1.T,D1.T)[:,L_ind_W]+np.kron(D1.T,D1.T)[:,U_ind_W]
    d1 = np.zeros((d1_D.shape[0],G+J))
    d1[:,:G] = d1_D
    d1[:,G:] = d1_L
    # d2 = np.kron(np.eye(G),D2.T)[:,L_ind_W]+np.kron(np.eye(G),D2.T)[:,U_ind_W]
    # d1 = np.kron(D1.T,D1.T)[:,L_ind_W]+np.kron(D1.T,D1.T)[:,U_ind_W]

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

    f=num_bins
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
    Sigma_L = [(np.kron(I_z[k],I_z[k])[:,L_ind_W] + np.kron(I_z[k],I_z[k])[:,U_ind_W])[L_ind_list[k]] for k in range(K)]
    Sigma_D = [(np.kron(I_z[k],I_z[k])[:,D_ind_W])[L_ind_list[k]] for k in range(K)]
    Sigma = np.zeros((np.sum(L),G+J))
    for k in range(K):
        Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],:G] = Sigma_D[k]
        Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],G:] = Sigma_L[k]
    # i_k = [(np.kron(I_z[k],I_z[k])[:,L_ind_W] + np.kron(I_z[k],I_z[k])[:,U_ind_W])[L_ind_list[k]] for k in range(K)]
    # Sigma = np.zeros((np.sum(L),J))
    # for k in range(K):
    #     Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],:] = i_k[k]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Estimated covariances
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    M_FULL = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        M_FULL[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Initialize

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    s_est = np.random.binomial(1,.5,np.sum(L))*1.
    p_est = s_est.copy()
    w_est = np.random.random(G+J)
    # w_est = np.random.random(J)
    v_est = w_est.copy()
    t_est = .5*(f@s_est + Sigma@w_est)
    u1 = np.zeros(np.sum(L))
    u2 = np.zeros(G+J)

    Phi_s1 = np.linalg.inv( theta*M.T@M + rho1*np.eye(np.sum(L)) + 2*Lmbda*f.T@f )
    Phi_w1 = np.linalg.inv( 4*lmbda*d2.T@d2 + 4*lmbda*d1.T@d1 + Lmbda*Sigma.T@Sigma + rho2*np.eye(G+J) )

    avg_time = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Estimation
    for admm_iter in range(num_admm_iters):

        # ----------------
        # s-update
        gamma = np.zeros(np.sum(L))
        gamma[(t_est>0)*(t_est<1)] = np.log(t_est[(t_est>0)*(t_est<1)]/(1-t_est[(t_est>0)*(t_est<1)]))
        Phi_s2 = rho1*(p_est-u1) + gamma + 2*Lmbda*f.T@t_est
        s_est = Phi_s1@Phi_s2

        # ----------------
        # p-update
        p_last=p_est.copy()
        ret = s_est+u1
        p_est = (ret-np.min(ret))/np.max((ret-np.min(ret)))
        # for k in range(K):
        #     p_est[k*L:(k+1)*L] = (ret[k*L:(k+1)*L] - np.min(ret[k*L:(k+1)*L])) / \
        #         np.max((ret[k*L:(k+1)*L] - np.min(ret[k*L:(k+1)*L])))
        p_est[p_est >  .5] = 1
        p_est[p_est <= .5] = 0

        # ----------------
        # t-update
        # # t_with_edge    = (1/4)*(f@s_est + Sigma@w_est) + (1/2)*np.sqrt((1/4)*(f@s_est + Sigma@w_est)**2 + (1/Lmbda))
        # # t_without_edge = (1/2)*(1 + (1/2)*(f@s_est + Sigma@w_est)) + (1/2)*np.sqrt( (1 - (1/2)*(f@s_est + Sigma@w_est))**2 + (1/Lmbda) )
        # # t_est[p_est==1] = t_with_edge[p_est==1]
        # # t_est[p_est==0] = t_without_edge[p_est==0]

        mu=pg_step_size
        t_hat = t_est.copy()
        t_hat[t_hat>1.]=1.
        t_hat[t_hat<0.]=0.
        ind_t_hat=(t_hat>0.)*(t_hat<1.)
        Update_Indices = np.ones(np.sum(L))==1
        for pg_iter in range(num_pg_iters):
            last_t=t_hat.copy()

            t_tild = last_t.copy()
            t_tild[ind_t_hat] = last_t[ind_t_hat] - mu*(s_est[ind_t_hat]-last_t[ind_t_hat])/(last_t[ind_t_hat]*(last_t[ind_t_hat]-1))
            t_hat = (1/(4*Lmbda+1/mu))*( 2*Lmbda*(f@s_est + Sigma@w_est) + (1/mu)*t_tild )
            t_hat[t_hat>1]=1
            t_hat[t_hat<0]=0
            ind_t_hat=(t_hat>0)*(t_hat<1)

            if np.linalg.norm(last_t-t_hat)**2<=pg_stop:
                break
        t_est = t_hat.copy()

        # ----------------
        # w-update
        Phi_w2 = Lmbda*Sigma.T@t_est + rho2*(v_est - u2)
        w_est = Phi_w1@Phi_w2

        # ----------------
        # v-update
        v_last = v_est.copy()
        v_est = w_est + u2
        v_est[v_est>1]=1
        v_est[v_est<0]=0

        # ----------------
        # u-updates
        u1 = u1 + s_est - p_est
        u2 = u2 + w_est - v_est

        # ----------------

        pr.append(
            np.linalg.norm(s_est-p_est,2) +
            np.linalg.norm(w_est-v_est,2)
            )
        dr.append(
            np.linalg.norm(rho1*(p_est-p_last),2) +
            np.linalg.norm(rho2*(v_est-v_last),2)
            )
        eps_pri.append(
            eps_abs*np.sqrt(len(s_est)) + eps_rel*np.amax( np.array([np.linalg.norm(s_est),np.linalg.norm(p_est)]) ) +
            eps_abs*np.sqrt(len(w_est)) + eps_rel*np.amax( np.array([np.linalg.norm(w_est),np.linalg.norm(v_est)]) )
            )
        eps_dua.append(
            eps_abs*np.sqrt(len(u1)) + eps_rel*(np.linalg.norm(rho1*u1)) +
            eps_abs*np.sqrt(len(u2)) + eps_rel*(np.linalg.norm(rho2*u2))
            )

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) or
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

        # End estimation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    w_ret = v_est.copy()
    s_ret = []
    t_ret = []
    for k in range(K):
        s_ret.append(p_est[np.sum(L[:k]):np.sum(L[:k])+L[k]])
        t_ret.append(t_est[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_ret, t_ret, w_ret

def stat_mod2(zeta_X,
              theta=10,rho1=10,rho2=10,phi=10,eta=1e-1,lmbda=1,Lmbda=10,
              eps_abs=0.,eps_rel=1e-3,
              num_bins=5,pg_step_size=.01,pg_stop=1e-2,
              num_admm_iters=1000,num_pg_iters=200,
              G=None
              ):
    K = len(zeta_X[0])
    N =  np.array([zeta_X[1][k].shape[0] for k in range(K)])
    if G is None:
        G = int(np.sum(N)+10)

    zeta_NUMBA = List()
    [zeta_NUMBA.append(zeta_X[0][k]) for k in range(K)]

    X_NUMBA = List()
    [X_NUMBA.append(zeta_X[1][k]) for k in range(K)]

    s_est, t_est, w_est = in_wrapper_stat_mod2(zeta_NUMBA,X_NUMBA,
                                               theta=theta,rho1=rho1,rho2=rho2,phi=phi,eta=eta,lmbda = lmbda,Lmbda = Lmbda,
                                               eps_abs = eps_abs,eps_rel = eps_rel,
                                               num_bins=num_bins,pg_step_size=pg_step_size,pg_stop=pg_stop,
                                               num_admm_iters=num_admm_iters,num_pg_iters=num_pg_iters,
                                               G=G
                                               )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est, t_est, w_est




























# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# sparse_stat_mod1
@jit(nopython=True)
def in_wrapper_sparse_stat_mod1(X,
                                        theta=10,rho1=10,rho2=10,rho3=10,phi=10,eta=1e-1,
                                        eps_abs=0.,eps_rel=1e-3,
                                        eps_t_thresh = 1e-6,
                                        num_admm_iters=1000,num_subadmm_iters=200,num_dca_iters=200
                                        ):
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

    R = (1/K)*np.kron(np.ones((1,K)),np.eye(L[0]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    M_FULL = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        M_FULL[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Q = np.zeros((K,np.sum(L)))
    for k in range(K):
        Q[k,np.sum(L[:k]):np.sum(L[:k])+L[k]]=1

    s_st = np.random.binomial(1,.5,np.sum(L))*1.
    p_st = s_st.copy()
    q_st = Q@s_st-1
    t_st = R@s_st
    u1 = np.zeros(np.sum(L))
    u2 = np.zeros(K)
    u3 = np.zeros(L[0])

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv( theta*M.T@M + (rho1+phi)*np.eye(np.sum(L)) + rho2*Q.T@Q + rho3*R.T@R )

    for admm_iter in range(num_admm_iters):
        # s-update
        s1 = s_st.copy()
        s2 = s_st.copy()
        w = np.zeros(np.sum(L))

        pr2 = []
        dr2 = []
        eps_pri2 = []
        eps_dua2 = []
        for admm_iter2 in range(num_subadmm_iters):
            s1 = soft_thresh(s2-w,2*eta/phi)
            Phi_s2 = rho1*(p_st-u1) + rho2*Q.T@(q_st+1-u2) + phi*(s1+w) + rho3*R.T@(t_st-u3)
            s2_last = s2.copy()
            s2 = Phi_s1@Phi_s2
            w = w + s1 - s2

            pr2.append(np.linalg.norm(s1-s2))
            dr2.append(np.linalg.norm(phi*(s2-s2_last)))
            eps_pri2.append( eps_abs*len(s1) + eps_rel*np.amax(np.array([np.linalg.norm(s1),np.linalg.norm(s2)])) )
            eps_dua2.append( eps_abs*len(s1) + eps_rel*np.linalg.norm(phi*w) )
            if (
                (np.sum(np.array(pr2)<=np.array(eps_pri2))>0) and
                (np.sum(np.array(dr2)<=np.array(eps_dua2))>0)
                ):
                break

        s_st = s2.copy()

        # p-update
        p_last=p_st.copy()
        ret = s_st + u1
        p_st = (ret-np.min(ret))/np.max((ret-np.min(ret)))
        p_st[p_st >= .5] = 1
        p_st[p_st <  .5] = 0

        # q-update
        q_last=q_st.copy()
        q_st = Q@s_st - 1 + u2
        q_st[q_st<0]=0

        # t-update
        t_last = t_st.copy()

        t_DCA = t_st.copy()
        y_DCA = np.zeros(L[0])
        currently_updating = np.ones(L[0])==1

        for idx in range(num_dca_iters):
            y_DCA[currently_updating] = 0
            y_DCA[currently_updating*(t_DCA>0)*(t_DCA<1)] = \
                K*np.log(t_DCA[currently_updating*(t_DCA>0)*(t_DCA<1)]/(1-t_DCA[currently_updating*(t_DCA>0)*(t_DCA<1)]))
            t_new = y_DCA/rho3 + R@s_st + u3
            t_new[t_new>1]=1
            t_new[t_new<0]=0

            currently_updating[np.abs(t_new-t_DCA) <= eps_t_thresh] = False
            t_DCA = t_new.copy()

            if np.sum(currently_updating) <= 0:
                break
        t_st = t_DCA.copy()

        # u-update
        u1 = u1 + s_st - p_st
        u2 = u2 + Q@s_st - q_st - 1
        u3 = u3 + R@s_st - t_st

        pr.append(
            np.linalg.norm(s_st-p_st,2) +
            np.linalg.norm(Q@s_st-q_st-1,2) +
            np.linalg.norm(R@s_st-t_st,2)
            )
        dr.append(
            np.linalg.norm(rho1*(p_st-p_last),2) +
            np.linalg.norm(rho2*Q.T@(q_st-q_last),2) +
            np.linalg.norm(rho3*R.T@(t_st-t_last),2)
            )
        eps_pri.append(
            eps_abs*np.sqrt(len(s_st)) + eps_rel*np.amax( np.array([np.linalg.norm(s_st),np.linalg.norm(p_st)]) ) +
            eps_abs*np.sqrt(len(s_st)) + eps_rel*np.amax( np.array([(np.linalg.norm(Q@s_st)),np.linalg.norm(q_st),np.linalg.norm(np.ones(K))]) ) +
            eps_abs*np.sqrt(len(t_st)) + eps_rel*np.amax( np.array([(np.linalg.norm(R@s_st)),np.linalg.norm(t_st)]) )
            )
        eps_dua.append(
            eps_abs*np.sqrt(len(u1)) + eps_rel*(np.linalg.norm(rho1*u1)) +
            eps_abs*np.sqrt(len(u2)) + eps_rel*(np.linalg.norm(rho2*Q.T@u2)) +
            eps_abs*np.sqrt(len(u3)) + eps_rel*(np.linalg.norm(rho3*R.T@u3))
            )

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) and
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

    s_ret = []
    t_ret = []
    for k in range(K):
        s_ret.append(p_st[np.sum(L[:k]):np.sum(L[:k])+L[k]])
        t_ret.append(t_st)

    return s_ret, t_ret

def sparse_stat_mod1(X,
                     theta=10,rho1=10,rho2=10,rho3=10,phi=10,eta=1e-1,
                     eps_abs=0.,eps_rel=1e-3,
                     eps_t_thresh = 1e-6,
                     num_admm_iters=1000,num_subadmm_iters=200,num_dca_iters=200
                     ):
    K = len(X)

    X_NUMBA = List()
    [X_NUMBA.append(X[k]) for k in range(K)]

    s_est,t_est = in_wrapper_sparse_stat_mod1(X_NUMBA,
                                              theta=theta,rho1=rho1,rho2=rho2,rho3=rho3,phi=phi,eta=eta,
                                              eps_abs=eps_abs,eps_rel=eps_rel,
                                              eps_t_thresh=eps_t_thresh,
                                              num_admm_iters=num_admm_iters,num_subadmm_iters=num_subadmm_iters,num_dca_iters=num_dca_iters
                                              )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est,t_est




























# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# sparse_stat_mod2
@jit(nopython=True)
def in_wrapper_sparse_stat_mod2(zeta, X,
                                        theta=10,rho1=10,rho2=10,rho3=10,phi=10,eta=1e-1,lmbda=1,Lmbda=10,
                                        eps_abs=0.,eps_rel=1e-3,
                                        num_bins=5,pg_step_size=.01,pg_stop=1e-2,
                                        num_admm_iters=1000,num_subadmm_iters=200,num_pg_iters=200,
                                        G=None
                                        ):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph parameters:
    K =  len(X)
    N =  np.array([X[k].shape[0] for k in range(K)])
    RK = np.array([X[k].shape[1] for k in range(K)])
    L = np.array([int(N[k]*(N[k]-1)/2) for k in range(K)])

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
    D_ind_W = np.arange(G)*(G+1)

    L_ind = np.zeros(np.sum(L))
    U_ind = np.zeros(np.sum(L))
    for k in range(K):
        L_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = L_ind_list[k] + np.sum(N[:k]**2)
        U_ind[np.sum(L[:k]):np.sum(L[:k])+L[k]] = U_ind_list[k] + np.sum(N[:k]**2)

    # Differential matrix for graphon
    D1=np.concatenate((-np.eye(G-1),np.zeros((1,G-1))),axis=0)+np.concatenate((np.zeros((1,G-1)),np.eye(G-1)),axis=0)
    D2=np.concatenate((np.eye(G-2),np.zeros((2,G-2))),axis=0)+np.concatenate((np.zeros((1,G-2)),np.concatenate((-2*np.eye(G-2),np.zeros((1,G-2))),axis=0)),axis=0)+np.concatenate((np.zeros((2,G-2)),np.eye(G-2)),axis=0)

    d2_D = np.kron(np.eye(G),D2.T)[:,D_ind_W]
    d2_L = np.kron(np.eye(G),D2.T)[:,L_ind_W]+np.kron(np.eye(G),D2.T)[:,U_ind_W]
    d2 = np.zeros((d2_D.shape[0],G+J))
    d2[:,:G] = d2_D
    d2[:,G:] = d2_L
    d1_D = np.kron(D1.T,D1.T)[:,D_ind_W]
    d1_L = np.kron(D1.T,D1.T)[:,L_ind_W]+np.kron(D1.T,D1.T)[:,U_ind_W]
    d1 = np.zeros((d1_D.shape[0],G+J))
    d1[:,:G] = d1_D
    d1[:,G:] = d1_L
    # d2 = np.kron(np.eye(G),D2.T)[:,L_ind_W]+np.kron(np.eye(G),D2.T)[:,U_ind_W]
    # d1 =np.kron(D1.T,D1.T)[:,L_ind_W]+np.kron(D1.T,D1.T)[:,U_ind_W]

    # Network histogram matrix for graphs
    f=num_bins
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
    Sigma_L = [(np.kron(I_z[k],I_z[k])[:,L_ind_W] + np.kron(I_z[k],I_z[k])[:,U_ind_W])[L_ind_list[k]] for k in range(K)]
    Sigma_D = [(np.kron(I_z[k],I_z[k])[:,D_ind_W])[L_ind_list[k]] for k in range(K)]
    Sigma = np.zeros((np.sum(L),G+J))
    for k in range(K):
        Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],:G] = Sigma_D[k]
        Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],G:] = Sigma_L[k]
    # i_k = [(np.kron(I_z[k],I_z[k])[:,L_ind_W] + np.kron(I_z[k],I_z[k])[:,U_ind_W])[L_ind_list[k]] for k in range(K)]
    # Sigma = np.zeros((np.sum(L),J))
    # for k in range(K):
    #     Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],:] = i_k[k]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    M_FULL = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        M_FULL[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Q = np.zeros((K,np.sum(L)))
    for k in range(K):
        Q[k,np.sum(L[:k]):np.sum(L[:k])+L[k]]=1

    s_st = np.random.binomial(1,.5,np.sum(L))*1.
    p_st = s_st.copy()
    q_st = Q@s_st-1
    w_st = np.random.random(G+J)
    v_st = w_st.copy()
    t_st = .5*(f@s_st + Sigma@w_st)
    u1 = np.zeros(np.sum(L))
    u2 = np.zeros(K)
    u3 = np.zeros(G+J)

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv( theta*M.T@M + (rho1+phi)*np.eye(np.sum(L)) + rho2*Q.T@Q + 2*Lmbda*f.T@f )
    Phi_w1 = np.linalg.inv( 4*lmbda*d2.T@d2 + 4*lmbda*d1.T@d1 + Lmbda*Sigma.T@Sigma + rho3*np.eye(G+J) )

    for admm_iter in range(num_admm_iters):
        # s-update
        s1 = s_st.copy()
        s2 = s_st.copy()
        v = np.zeros(np.sum(L))
        gamma = np.zeros(np.sum(L))
        gamma[(t_st>0)*(t_st<1)] = np.log(t_st[(t_st>0)*(t_st<1)]/(1-t_st[(t_st>0)*(t_st<1)]))

        pr2 = []
        dr2 = []
        eps_pri2 = []
        eps_dua2 = []
        for admm_iter2 in range(num_subadmm_iters):
            s1 = soft_thresh(s2-v,2*eta/phi)
            Phi_s2 = rho1*(p_st-u1) + rho2*Q.T@(q_st+1-u2) + phi*(s1+v) + gamma + 2*Lmbda*f.T@t_st
            s2_last = s2.copy()
            s2 = Phi_s1@Phi_s2
            v = v + s1 - s2

            pr2.append(np.linalg.norm(s1-s2))
            dr2.append(np.linalg.norm(phi*(s2-s2_last)))
            eps_pri2.append( eps_abs*len(s1) + eps_rel*np.amax(np.array([np.linalg.norm(s1),np.linalg.norm(s2)])) )
            eps_dua2.append( eps_abs*len(s1) + eps_rel*np.linalg.norm(phi*v) )
            if (
                (np.sum(np.array(pr2)<=np.array(eps_pri2))>0) and
                (np.sum(np.array(dr2)<=np.array(eps_dua2))>0)
                ):
                break

        s_st = s2.copy()

        # p-update
        p_last=p_st.copy()
        ret = s_st + u1
        p_st = (ret-np.min(ret))/np.max((ret-np.min(ret)))
        p_st[p_st >= .5] = 1
        p_st[p_st <  .5] = 0

        # q-update
        q_last=q_st.copy()
        q_st = Q@s_st - 1 + u2
        q_st[q_st<0]=0

        mu=pg_step_size
        t_hat = t_st.copy()
        t_hat[t_hat>1.]=1.
        t_hat[t_hat<0.]=0.
        ind_t_hat=(t_hat>0.)*(t_hat<1.)
        Update_Indices = np.ones(np.sum(L))==1
        for pg_iter in range(num_pg_iters):
            last_t=t_hat.copy()

            t_tild = last_t.copy()
            t_tild[ind_t_hat] = last_t[ind_t_hat] - mu*(s_st[ind_t_hat]-last_t[ind_t_hat])/(last_t[ind_t_hat]*(last_t[ind_t_hat]-1))
            t_hat = (1/(4*Lmbda+1/mu))*( 2*Lmbda*(f@s_st + Sigma@w_st) + (1/mu)*t_tild )
            t_hat[t_hat>1]=1
            t_hat[t_hat<0]=0
            ind_t_hat=(t_hat>0)*(t_hat<1)

            if np.linalg.norm(last_t-t_hat)**2<=pg_stop:
                break
        t_st = t_hat.copy()

        # w-update
        Phi_w2 = Lmbda*Sigma.T@t_st + rho3*(v_st - u3)
        w_st = Phi_w1@Phi_w2

        # v-update
        v_last = v_st.copy()
        v_st = w_st + u3
        v_st[v_st>1]=1
        v_st[v_st<0]=0

        # u-update
        u1 = u1 + s_st - p_st
        u2 = u2 + Q@s_st - q_st - 1
        u3 = u3 + w_st - v_st

        pr.append(
            np.linalg.norm(s_st-p_st,2) +
            np.linalg.norm(Q@s_st-q_st-1,2) +
            np.linalg.norm(w_st-v_st,2)
            )
        dr.append(
            np.linalg.norm(rho1*(p_st-p_last),2) +
            np.linalg.norm(rho2*Q.T@(q_st-q_last),2) +
            np.linalg.norm(rho3*(v_st-v_last),2)
            )
        eps_pri.append(
            eps_abs*np.sqrt(len(s_st)) + eps_rel*np.amax( np.array([np.linalg.norm(s_st),np.linalg.norm(p_st)]) ) +
            eps_abs*np.sqrt(len(s_st)) + eps_rel*np.amax( np.array([(np.linalg.norm(Q@s_st)),np.linalg.norm(q_st),np.linalg.norm(np.ones(K))]) ) +
            eps_abs*np.sqrt(len(w_st)) + eps_rel*np.amax( np.array([(np.linalg.norm(w_st)),np.linalg.norm(v_st)]) )
            )
        eps_dua.append(
            eps_abs*np.sqrt(len(u1)) + eps_rel*(np.linalg.norm(rho1*u1)) +
            eps_abs*np.sqrt(len(u2)) + eps_rel*(np.linalg.norm(rho2*Q.T@u2)) +
            eps_abs*np.sqrt(len(u3)) + eps_rel*(np.linalg.norm(rho3*u3))
            )

        if (
            (np.sum(np.array(pr)<=np.array(eps_pri))>0) and
            (np.sum(np.array(dr)<=np.array(eps_dua))>0)
            ):
            break

    s_ret = []
    t_ret = []
    for k in range(K):
        s_ret.append(p_st[np.sum(L[:k]):np.sum(L[:k])+L[k]])
        t_ret.append(t_st[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_ret, t_ret, v_st

def sparse_stat_mod2(zeta_X,
                     theta=10,rho1=10,rho2=10,rho3=10,phi=10,eta=1e-1,lmbda=1,Lmbda=10,
                     eps_abs=0.,eps_rel=1e-3,
                     num_bins=5,pg_step_size=.01,pg_stop=1e-2,
                     num_admm_iters=1000,num_subadmm_iters=200,num_pg_iters=200,
                     G=None
                     ):
    K = len(zeta_X[0])
    N =  np.array([zeta_X[1][k].shape[0] for k in range(K)])
    if G is None:
        G = int(np.sum(N)+10)

    zeta_NUMBA = List()
    [zeta_NUMBA.append(zeta_X[0][k]) for k in range(K)]

    X_NUMBA = List()
    [X_NUMBA.append(zeta_X[1][k]) for k in range(K)]

    s_est, t_est, w_est = in_wrapper_sparse_stat_mod2(zeta_NUMBA,X_NUMBA,
                                                      theta=theta,rho1=rho1,rho2=rho2,rho3=rho3,phi=phi,eta=eta,lmbda=lmbda,Lmbda=Lmbda,
                                                      eps_abs=eps_abs,eps_rel=eps_rel,
                                                      num_bins=num_bins,pg_step_size=pg_step_size,pg_stop=pg_stop,
                                                      num_admm_iters=num_admm_iters,num_subadmm_iters=num_subadmm_iters,num_pg_iters=num_pg_iters,
                                                      G=G
                                                      )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est, t_est, w_est




























# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# pairdiff_stat
@jit(nopython=True)
def in_wrapper_pairdiff_stat(X,
                             theta=10,rho1=10,rho2=10,eta=1e-1,phi=10,
                             eps_abs=0.,eps_rel=1e-3,
                             num_admm_iters=1000
                             ):
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
    M_FULL = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        M_FULL[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    s_pd = np.random.binomial(1,.5,np.sum(L))*1.
    p_pd = s_pd.copy()
    l_pd = W@s_pd
    u1 = np.zeros(int(K*L[0]*(K-1)/2))
    u2 = np.zeros(np.sum(L))


    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv(theta*M.T@M + rho1*W.T@W + rho2*np.eye(np.sum(L)))

    for admm_iter in range(num_admm_iters):
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

    s_ret = []
    for k in range(K):
        s_ret.append(p_pd[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_ret

def pairdiff_stat(X,
                  theta=10,rho1=10,rho2=10,eta=1e-1,phi=10,
                  eps_abs=0.,eps_rel=1e-3,
                  num_admm_iters=1000
                  ):
    K = len(X)

    X_NUMBA = List()
    [X_NUMBA.append(X[k]) for k in range(K)]

    s_est = in_wrapper_pairdiff_stat(X_NUMBA,
                                     theta=theta,rho1=rho1,rho2=rho2,eta=eta,phi=phi,
                                     eps_abs=eps_abs,eps_rel=eps_rel,
                                     num_admm_iters=num_admm_iters
                                     )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est




























# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# pairdiff_stat_mod1
@jit(nopython=True)
def in_wrapper_pairdiff_stat_mod1(X,
                                  theta=10,rho1=10,rho2=10,rho3=10,eta=1e-1,phi=10,
                                  eps_abs=0.,eps_rel=1e-3,
                                  eps_t_thresh = 1e-6,
                                  num_admm_iters=1000,num_dca_iters=200
                                  ):
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
    M_FULL = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        M_FULL[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    s_pd = np.random.binomial(1,.5,np.sum(L))*1.
    p_pd = s_pd.copy()
    l_pd = W@s_pd
    t_pd = R@s_pd
    u1 = np.zeros(int(K*L[0]*(K-1)/2))
    u2 = np.zeros(np.sum(L))
    u3 = np.zeros(L[0])

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv(theta*M.T@M + rho1*W.T@W + rho2*np.eye(np.sum(L)) + rho3*R.T@R)

    for admm_iter in range(num_admm_iters):
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
        t_last = t_pd.copy()

        t_DCA = t_pd.copy()
        y_DCA = np.zeros(L[0])
        currently_updating = np.ones(L[0])==1

        for idx in range(num_dca_iters):
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

    s_ret = []
    t_ret = []
    for k in range(K):
        s_ret.append(p_pd[np.sum(L[:k]):np.sum(L[:k])+L[k]])
        t_ret.append(t_pd)

    return s_ret, t_ret

def pairdiff_stat_mod1(X,
                       theta=10,rho1=10,rho2=10,rho3=10,eta=1e-1,phi=10,
                       eps_abs=0.,eps_rel=1e-3,
                       eps_t_thresh = 1e-6,
                       num_admm_iters=1000,num_dca_iters=200
                       ):
    K = len(X)

    X_NUMBA = List()
    [X_NUMBA.append(X[k]) for k in range(K)]

    s_est, t_est = in_wrapper_pairdiff_stat_mod1(X_NUMBA,
                                                 theta=theta,rho1=rho1,rho2=rho2,rho3=rho3,eta=eta,phi=phi,
                                                 eps_abs=eps_abs,eps_rel=eps_rel,
                                                 eps_t_thresh=eps_t_thresh,
                                                 num_admm_iters=num_admm_iters,num_dca_iters=num_dca_iters
                                                 )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est, t_est




























# ##############################################################################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# pairdiff_stat_mod2
@jit(nopython=True)
def in_wrapper_pairdiff_stat_mod2(zeta,X,
                                  theta=10,rho1=10,rho2=10,rho3=10,eta=1e-1,phi=10,lmbda=1,Lmbda=10,
                                  eps_abs=0.,eps_rel=1e-3,
                                  num_bins=5,pg_step_size=.01,pg_stop=1e-2,
                                  num_admm_iters=1000,num_pg_iters=200,
                                  G=None
                                  ):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph parameters:
    K =  len(X)
    N =  np.array([X[k].shape[0] for k in range(K)])
    RK = np.array([X[k].shape[1] for k in range(K)])
    L = np.array([int(N[k]*(N[k]-1)/2) for k in range(K)])

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
    D_ind_W = np.arange(G)*(G+1)

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

    d2_D = np.kron(np.eye(G),D2.T)[:,D_ind_W]
    d2_L = np.kron(np.eye(G),D2.T)[:,L_ind_W]+np.kron(np.eye(G),D2.T)[:,U_ind_W]
    d2 = np.zeros((d2_D.shape[0],G+J))
    d2[:,:G] = d2_D
    d2[:,G:] = d2_L
    d1_D = np.kron(D1.T,D1.T)[:,D_ind_W]
    d1_L = np.kron(D1.T,D1.T)[:,L_ind_W]+np.kron(D1.T,D1.T)[:,U_ind_W]
    d1 = np.zeros((d1_D.shape[0],G+J))
    d1[:,:G] = d1_D
    d1[:,G:] = d1_L
    # d2 = np.kron(np.eye(G),D2.T)[:,L_ind_W]+np.kron(np.eye(G),D2.T)[:,U_ind_W]
    # d1 =np.kron(D1.T,D1.T)[:,L_ind_W]+np.kron(D1.T,D1.T)[:,U_ind_W]

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

    f=num_bins
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
    Sigma_L = [(np.kron(I_z[k],I_z[k])[:,L_ind_W] + np.kron(I_z[k],I_z[k])[:,U_ind_W])[L_ind_list[k]] for k in range(K)]
    Sigma_D = [(np.kron(I_z[k],I_z[k])[:,D_ind_W])[L_ind_list[k]] for k in range(K)]
    Sigma = np.zeros((np.sum(L),G+J))
    for k in range(K):
        Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],:G] = Sigma_D[k]
        Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],G:] = Sigma_L[k]
    # i_k = [(np.kron(I_z[k],I_z[k])[:,L_ind_W] + np.kron(I_z[k],I_z[k])[:,U_ind_W])[L_ind_list[k]] for k in range(K)]
    # Sigma = np.zeros((np.sum(L),J))
    # for k in range(K):
    #     Sigma[np.sum(L[:k]):np.sum(L[:k])+L[k],:] = i_k[k]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C = []
    [C.append((X[k]@X[k].T)/RK[k]) for k in range(K)]
    M_FULL = np.zeros((np.sum(N**2),np.sum(N**2)))
    for k in range(K):
        M_FULL[
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k],
              np.sum((N**2)[:k]):np.sum((N**2)[:k])+(N**2)[k]
              ] \
            += np.kron(-C[k],np.eye(N[k])) + np.kron(np.eye(N[k]), C[k])
    M = np.zeros((np.sum(N**2),np.sum(L)))
    for kl in range(np.sum(L)):
        M[:,kl] = M_FULL[:,int(L_ind[kl])]+M_FULL[:,int(U_ind[kl])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    s_pd = np.random.binomial(1,.5,np.sum(L))*1.
    p_pd = s_pd.copy()
    l_pd = W@s_pd
    w_pd = np.random.random(G+J)
    v_pd = w_pd.copy()
    t_pd = .5*(f@s_pd + Sigma@w_pd)

    u1 = np.zeros(int(K*L[0]*(K-1)/2))
    u2 = np.zeros(np.sum(L))
    u3 = np.zeros(G+J)

    pr=[]
    dr=[]
    eps_pri=[]
    eps_dua=[]

    def soft_thresh(z, a):
        return (z-a)*(z-a > 0) - (-z-a)*(-z-a > 0)

    Phi_s1 = np.linalg.inv(theta*M.T@M + rho1*W.T@W + rho2*np.eye(np.sum(L)) + 2*Lmbda*f.T@f)
    Phi_w1 = np.linalg.inv( 4*lmbda*d2.T@d2 + 4*lmbda*d1.T@d1 + 2*Lmbda*Sigma.T@Sigma + rho3*np.eye(G+J) )

    for admm_iter in range(num_admm_iters):
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

        mu=pg_step_size
        t_hat = t_pd.copy()
        t_hat[t_hat>1.]=1.
        t_hat[t_hat<0.]=0.
        ind_t_hat=(t_hat>0.)*(t_hat<1.)
        Update_Indices = np.ones(np.sum(L))==1
        for pg_iter in range(num_pg_iters):
            last_t=t_hat.copy()

            t_tild = last_t.copy()
            t_tild[ind_t_hat] = last_t[ind_t_hat] - mu*(s_pd[ind_t_hat]-last_t[ind_t_hat])/(last_t[ind_t_hat]*(last_t[ind_t_hat]-1))
            t_hat = (1/(4*Lmbda+1/mu))*( 2*Lmbda*(f@s_pd + Sigma@w_pd) + (1/mu)*t_tild )
            t_hat[t_hat>1]=1
            t_hat[t_hat<0]=0
            ind_t_hat=(t_hat>0)*(t_hat<1)

            if np.linalg.norm(last_t-t_hat)**2<=pg_stop:
                break
        t_pd = t_hat.copy()

        # w-update
        Phi_w2 = Lmbda*Sigma.T@t_pd + rho3*(v_pd - u3)
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

    s_ret = []
    t_ret = []
    for k in range(K):
        s_ret.append(p_pd[np.sum(L[:k]):np.sum(L[:k])+L[k]])
        t_ret.append(t_pd[np.sum(L[:k]):np.sum(L[:k])+L[k]])

    return s_ret, t_ret, v_pd

def pairdiff_stat_mod2(zeta_X,
                       theta=10,rho1=10,rho2=10,rho3=10,eta=1e-1,phi=10,lmbda=1,Lmbda=10,
                       eps_abs=0.,eps_rel=1e-3,
                       num_bins=5,pg_step_size=.01,pg_stop=1e-2,
                       num_admm_iters=1000,num_pg_iters=200,
                       G=None
                       ):
    K = len(zeta_X[0])
    N =  np.array([zeta_X[1][k].shape[0] for k in range(K)])
    if G is None:
        G = int(np.sum(N)+10)

    zeta_NUMBA = List()
    [zeta_NUMBA.append(zeta_X[0][k]) for k in range(K)]

    X_NUMBA = List()
    [X_NUMBA.append(zeta_X[1][k]) for k in range(K)]

    s_est,t_est,w_est = in_wrapper_pairdiff_stat_mod2(zeta_NUMBA,X_NUMBA,
                                                      theta=theta,rho1=rho1,rho2=rho2,rho3=rho3,eta=eta,phi=phi,lmbda=lmbda,Lmbda=Lmbda,
                                                      eps_abs=eps_abs,eps_rel=eps_rel,
                                                      pg_step_size=pg_step_size,pg_stop=pg_stop,
                                                      num_admm_iters=num_admm_iters,num_pg_iters=num_pg_iters,
                                                      G=G
                                                      )
    for k in range(K):
        s_est[k] = s_est[k].astype(int)
    return s_est,t_est,w_est
