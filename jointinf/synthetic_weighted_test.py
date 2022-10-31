# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from helper import *
from methods_weighted import *

import time
import os
import csv
import datetime

from numba import jit,njit,prange
from numba.typed import List
import numba as nb
from numba import cuda

# Set up folder for saving simulations
experiment_type = 'edgeweight'

date_str = datetime.datetime.today().strftime('%Y%m%d')
path = os.getcwd()
folder_path = path+'/src/'+'EXP_'+date_str+'_'+experiment_type
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
randomseed=1000
np.random.seed(randomseed)

status_file_name = path+'/src/'+'EXP_'+date_str+'_'+experiment_type+'/status.txt'
exp_status_file = open(status_file_name,'w')
exp_status_file.writelines('Experiment (type: '+experiment_type+') on date: '+date_str+'\n')
exp_status_file.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#-------------------------
graphon_fam = 'quad'
nodeset = 'samenodeset'
sigmodel = 'statsig'

# Other possibilities: smoothsig, mrfsig, other graphon_fam
# Other possibilities: samenodeset, diffsetsamesize, diffsize
# synth remains the same for this framework
exp_status_file = open(status_file_name,'a')
exp_status_file.write('Node sets: '+nodeset + '\n')
exp_status_file.close()

experiment_name = 'EXP_' + experiment_type + '_' + graphon_fam + '_' + nodeset + '_'+sigmodel
if not os.path.exists(folder_path+'/'+experiment_name):
    os.makedirs(folder_path+'/'+experiment_name)
#-------------------------

#-------------------------
method_names = ['sparse_stat',
                'stat_mod1','stat_mod2',
                'sparse_stat_mod1','sparse_stat_mod2',
                'pairdiff_stat',
                'pairdiff_stat_mod1','pairdiff_stat_mod2'
                ]
num_methods = len(method_names)
method_vec = np.ones(num_methods,dtype=int)
method_vec = np.array([1,1,1,0,0,1,1,1])
if nodeset=='diffsize':
    method_vec[1] = 0
    method_vec[3] = 0
    method_vec[5] = 0
    method_vec[6] = 0
    method_vec[7] = 0
#-------------------------

#-------------------------------------------------------------------------------
# Simulation setting

#-------------------------
num_graph_trials = 10
#-------------------------

#-------------------------
signal_range = (np.logspace(2,5,4)).astype(int)
len_signal_range = len(signal_range)
#-------------------------

#-------------------------
X_SET = []
X_MRF_SET = []
X_smooth_SET = []

zeta_SET = []

s_TRUE = []
t_TRUE = []
w_TRUE = []
#-------------------------

#-------------------------------------------------------------------------------
# Problem parameters

#-------------------------
K = 3
N = np.array([30,15,45])
if (nodeset=='samenodeset') or (nodeset=='samesizediffset'):
    N = N[0]*np.ones(K,dtype=int)
L = (N*(N-1)/2).astype(int)
G = np.sum(N)+10
J = int(G*(G-1)/2)
#-------------------------

#-------------------------
graphon_inds=(np.arange(G+1)/(G+1))[1:]
W=graphon(graphon_inds.reshape(-1,1),graphon_inds.reshape(1,-1),graphon_fam=graphon_fam)
w=np.concatenate((W[np.eye(G)==1],mat2lowtri(W)))

w_TRUE = w.copy()
#-------------------------

#-------------------------
Filt_Order = 3
h = np.random.normal(0,1,Filt_Order)
# h = np.random.random(F)

alpha = 5*np.random.random()
beta = np.random.random()
#-------------------------

#-------------------------------------------------------------------------------
# Trials of sampling graphs from graphon

exp_status_file = open(status_file_name,'a')
exp_status_file.write('Generating graph signals...' + '\n')
exp_status_file.close()
print('Generating graph signals...')

for graph_trial in range(num_graph_trials):
    exp_status_file = open(status_file_name,'a')
    exp_status_file.write(f'  Trial {graph_trial+1}' + '\n')
    exp_status_file.close()
    print(f'  Trial {graph_trial+1}')

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
    S[0][np.tril(np.ones((N[0],N[0])))-np.eye(N[0])==1] = np.random.binomial(1,T[0][np.tril(np.ones((N[0],N[0])))-np.eye(N[0]) == 1])
    S[0] *= np.random.random((N[0],N[0]))
    S[0] += S[0].T
    for k in range(1,K):
        if nodeset=='samenodeset':
            zeta.append(zeta[0])
            T.append(T[0])
        else:
            zeta.append(np.sort(np.random.random(N[k])))
            T.append(graphon(zeta[k].reshape(-1,1),zeta[k].reshape(1,-1),graphon_fam=graphon_fam))
        S.append(np.zeros((N[k],N[k])))
        S[k][np.tril(np.ones((N[k],N[k])))-np.eye(N[k])==1] = np.random.binomial(1, T[k][np.tril(np.ones((N[k],N[k])))-np.eye(N[k]) == 1])
        S[k] *= np.random.random((N[k],N[k]))
        S[k] += S[k].T
    #-------------------------

    #-------------------------
    s = []
    t = []
    for k in range(K):
        s.append(mat2lowtri(S[k]))
        t.append(mat2lowtri(T[k]))
    s_TRUE.append(s)
    t_TRUE.append(t)

    z=[]
    for k in range(K):
        z.append(np.floor(zeta[k]*G).astype(int))
        z[k][z[k]>=G]=G-1
    #-------------------------

    #-------------------------
    L_ind_W=np.squeeze(np.where(np.matrix.flatten(np.tril(np.ones((G,G)))-np.eye(G),'F')))
    I_z = []
    for k in range(K):
        zeta_idx=np.squeeze(np.where(np.triu(np.ones((N[k],N[k])))-np.eye(N[k])==1))
        I_z.append((np.eye(G**2)[(z[k][zeta_idx[0]])*G+z[k][zeta_idx[1]],:]+np.eye(G**2)[(z[k][zeta_idx[1]])*G+z[k][zeta_idx[0]],:])[:,L_ind_W])
    #-------------------------

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

    #-------------------------
    X_MAX = []
    X_MRF_MAX = []
    X_smooth_MAX = []
    for k in range(K):
        X_MAX.append(H[k]@np.random.normal(0,1,(N[k],signal_range[-1])))
        X_MRF_MAX.append(H_MRF[k]@np.random.normal(0,1,(N[k],signal_range[-1])))
        X_smooth_MAX.append(H_smooth[k]@np.random.normal(0,1,(N[k],signal_range[-1])))
    #-------------------------

    X_SET[graph_trial] = X_MAX
    X_MRF_SET[graph_trial] = X_MRF_MAX
    X_smooth_SET[graph_trial] = X_smooth_MAX
    zeta_SET[graph_trial] = zeta

print('Generated graph signals')
exp_status_file = open(status_file_name,'a')
exp_status_file.write('Generated graph signals' + '\n')
exp_status_file.close()

if not os.path.exists(folder_path+'/'+experiment_name+'/data'):
    os.makedirs(folder_path+'/'+experiment_name+'/data')

for graph_trial in range(num_graph_trials):
    with open(folder_path+'/'+experiment_name+'/data/'+
              'X'+'_trial'+str(graph_trial)+'.csv','w') as f:
        writer = csv.writer(f,delimiter=',')
        for k in range(K):
            for i in range(N[k]):
                writer.writerow(X_SET[graph_trial][k][i,:].reshape(-1))
    with open(folder_path+'/'+experiment_name+'/data/'+
              'zeta'+'_trial'+str(graph_trial)+'.csv','w') as f:
        writer = csv.writer(f,delimiter=',')
        for k in range(K):
            writer.writerow(zeta_SET[graph_trial][k].reshape(-1))

# sims_params.csv:
#   First row:
#       num_graph_trials: no. trials
#       K: no. graphs
#       G: nodes of discrete graphon
#       N: vector of nodes per graph
#       graphon_fam: Type of graphon used to generate
#   Second row:
#       signal_range: vector of observed signals
#   Third row:
#       Methods used:
#           Column 1: sparse_stat
#           Column 2: stat_mod1
#           Column 3: stat_mod2
#           Column 4: sparse_stat_mod1
#           Column 5: sparse_stat_mod2
#           Column 6: pairdiff_stat
#           Column 7: pairdiff_stat_mod1
#           Column 8: pairdiff_stat_mod2
with open(folder_path+'/'+experiment_name+'/'+'sim_params'+'.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(np.concatenate(([num_graph_trials,K,G],N)).reshape(-1))
    writer.writerow(signal_range.reshape(-1))
    writer.writerow(method_vec.reshape(-1))

# truth.csv:
#   1st num_graph_trials rows:
#       Row is concatenation of lower triangle of true s
#   2nd num_graph_trials rows:
#       Row is concatenation of lower triangle of true t
#   Last row:
#       Row is lower triangle of true w
with open(folder_path+'/'+experiment_name+'/'+
          'truth'+'.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for graph_trial in range(num_graph_trials):
        writer.writerow(np.concatenate(s_TRUE[graph_trial]))
    for graph_trial in range(num_graph_trials):
        writer.writerow(np.concatenate(t_TRUE[graph_trial]))
    writer.writerow(w_TRUE)






# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-------------------------------------------------------------------------------
# Estimate graphs

#-------------------------
s_EST    = [[[[] for graph_trial in range(num_graph_trials)] for num_sig_trial in range(len_signal_range)] for method_iter in range(num_methods)]
sbar_EST = [[[[] for graph_trial in range(num_graph_trials)] for num_sig_trial in range(len_signal_range)] for method_iter in range(num_methods)]
t_EST    = [[[[] for graph_trial in range(num_graph_trials)] for num_sig_trial in range(len_signal_range)] for method_iter in range(num_methods)]
w_EST    = [[[[] for graph_trial in range(num_graph_trials)] for num_sig_trial in range(len_signal_range)] for method_iter in range(num_methods)]
s_ERROR    = [-np.ones((num_graph_trials,len_signal_range),dtype=int) for method_iter in range(num_methods)]
sbar_ERROR = [-np.ones((num_graph_trials,len_signal_range),dtype=int) for method_iter in range(num_methods)]
t_ERROR    = [-np.ones((num_graph_trials,len_signal_range),dtype=int) for method_iter in range(num_methods)]
w_ERROR    = [-np.ones((num_graph_trials,len_signal_range),dtype=int) for method_iter in range(num_methods)]
time_method = [np.zeros((num_graph_trials,len_signal_range)) for method_iter in range(num_methods)]

for method_iter in range(num_methods):
    if method_vec[method_iter]:
        s_ERROR[method_iter] = np.zeros((num_graph_trials,len_signal_range))
        sbar_ERROR[method_iter] = np.zeros((num_graph_trials,len_signal_range))
        if any(method_iter==np.array([1,2,3,4,6,7])):
            t_ERROR[method_iter] = np.zeros((num_graph_trials,len_signal_range))
        else:
            t_ERROR[method_iter] = -np.ones((num_graph_trials,len_signal_range),dtype=int)
        if any(method_iter==np.array([2,4,7])):
            w_ERROR[method_iter] = np.zeros((num_graph_trials,len_signal_range))
        else:
            w_ERROR[method_iter] = -np.ones((num_graph_trials,len_signal_range),dtype=int)
        time_method[method_iter] = np.zeros((num_graph_trials,len_signal_range))
#-------------------------

#---------------------------------------------------------------------------
for graph_trial in range(num_graph_trials):
    #-------------------------
    print(f'Trial {graph_trial+1}')
    exp_status_file = open(status_file_name,'a')
    exp_status_file.write(f'Trial {graph_trial+1}' + '\n')
    exp_status_file.close()
    #-------------------------

    #-------------------------------------------------------------------------------
    for num_sig_trial in range(len_signal_range):
        #-------------------------
        num_signals = signal_range[num_sig_trial]

        print(f'  No. of signals: {num_signals}')
        exp_status_file = open(status_file_name,'a')
        exp_status_file.write(f'  No. of signals: {num_signals}' + '\n')
        exp_status_file.close()

        X_curr = [X_SET[graph_trial][k][:,:num_signals] for k in range(K)]
        zeta_X_curr = (zeta_SET[graph_trial],X_curr)
        print('')
        #-------------------------

        #-------------------------
        for method_iter in range(num_methods):
            if method_vec[method_iter]:
                print('    Method: '+method_names[method_iter])
                exp_status_file = open(status_file_name,'a')
                exp_status_file.write('    Method: '+method_names[method_iter] + '\n')
                exp_status_file.close()

                tic = time.perf_counter()
                if method_iter==0:
                    s_curr,sbar_curr = sparse_stat(X_curr,pg_stepsize=1e-4)
                    if s_curr[0][0]!=0 and s_curr[0][0]!=1:
                        for k in range(K):
                            s_curr[k] = (sbar_curr[k]>0).astype(int)
                elif method_iter==1:
                    s_curr,sbar_curr,t_curr = stat_mod1(X_curr)
                elif method_iter==2:
                    s_curr,sbar_curr,t_curr,w_curr = stat_mod2(zeta_X_curr)
                elif method_iter==3:
                    s_curr,sbar_curr,t_curr = sparse_stat_mod1(X_curr)
                elif method_iter==4:
                    s_curr,sbar_curr,t_curr,w_curr = sparse_stat_mod2(zeta_X_curr)
                elif method_iter==5:
                    s_curr,sbar_curr = pairdiff_stat(X_curr)
                elif method_iter==6:
                    s_curr,sbar_curr,t_curr = pairdiff_stat_mod1(X_curr)
                elif method_iter==7:
                    s_curr,sbar_curr,t_curr,w_curr = pairdiff_stat_mod2(zeta_X_curr)
                else:
                    s_curr = -np.ones(np.sum(L),dtype=int)
                    sbar_curr = -np.ones(np.sum(L),dtype=int)

                toc = time.perf_counter()
                time_method[method_iter][graph_trial,num_sig_trial] = np.abs(toc-tic)
                if np.abs(toc-tic)<60:
                    print(f'      Time: {np.abs(toc-tic):.2f} s')
                    exp_status_file = open(status_file_name,'a')
                    exp_status_file.write(f'      Time: {np.abs(toc-tic):.2f} s' + '\n')
                    exp_status_file.close()
                elif np.abs(toc-tic)<3600:
                    print(f'      Time: {np.abs(toc-tic)/60:.2f} min')
                    exp_status_file = open(status_file_name,'a')
                    exp_status_file.write(f'      Time: {np.abs(toc-tic)/60:.2f} min' + '\n')
                    exp_status_file.close()
                else:
                    print(f'      Time: {np.abs(toc-tic)/3600:.2f} hr')
                    exp_status_file = open(status_file_name,'a')
                    exp_status_file.write(f'      Time: {np.abs(toc-tic)/3600:.2f} hr' + '\n')
                    exp_status_file.close()

                s_EST[method_iter][num_sig_trial][graph_trial] = s_curr.copy()
                s_BIN = [(s_TRUE[graph_trial][k]>0).astype(int) for k in range(K)]
                for k in range(K):
                    if np.max(s_curr[k])<1e-9 and np.max(s_BIN[k])<1e-9:
                        s_ERROR[method_iter][graph_trial,num_sig_trial] += 0
                    elif np.max(s_curr[k])<1e-9:
                        s_ERROR[method_iter][graph_trial,num_sig_trial] += np.linalg.norm(s_curr[k]-s_TRUE[graph_trial][k],1)/np.sum(L)
                    elif np.max(s_TRUE[graph_trial][k])<1e-9:
                        s_ERROR[method_iter][graph_trial,num_sig_trial] += np.linalg.norm(s_curr[k]-s_TRUE[graph_trial][k],1)/np.sum(L)
                    else:
                        s_ERROR[method_iter][graph_trial,num_sig_trial] += np.linalg.norm(s_curr[k]-s_TRUE[graph_trial][k],1)/np.sum(L)

                sbar_EST[method_iter][num_sig_trial][graph_trial] = sbar_curr.copy()
                for k in range(K):
                    if np.max(sbar_curr[k])<1e-9 and np.max(s_TRUE[graph_trial][k])<1e-9:
                        sbar_ERROR[method_iter][graph_trial,num_sig_trial] += 0
                    elif np.max(sbar_curr[k])<1e-9:
                        sbar_ERROR[method_iter][graph_trial,num_sig_trial] += np.linalg.norm(sbar_curr[k]-s_TRUE[graph_trial][k]/np.max(s_TRUE[graph_trial][k]),1)/np.sum(L)
                    elif np.max(s_TRUE[graph_trial][k])<1e-9:
                        sbar_ERROR[method_iter][graph_trial,num_sig_trial] += np.linalg.norm(sbar_curr[k]/np.max(sbar_curr[k])-s_TRUE[graph_trial][k],1)/np.sum(L)
                    else:
                        sbar_ERROR[method_iter][graph_trial,num_sig_trial] += np.linalg.norm(sbar_curr[k]/np.max(sbar_curr[k])-s_TRUE[graph_trial][k]/np.max(s_TRUE[graph_trial][k]),1)/np.sum(L)
                print(f'      Error s: {s_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')')
                exp_status_file = open(status_file_name,'a')
                exp_status_file.write(f'      Error s: {s_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')' + '\n')
                exp_status_file.close()

                print(f'      Error sbar: {sbar_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')')
                exp_status_file = open(status_file_name,'a')
                exp_status_file.write(f'      Error sbar: {sbar_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')' + '\n')
                exp_status_file.close()

                if any(method_iter==np.array([1,2,3,4,6,7])):
                    t_EST[method_iter][num_sig_trial][graph_trial] = t_curr.copy()
                    t_ERROR[method_iter][graph_trial,num_sig_trial] = np.sum([np.linalg.norm(t_curr[k]-t_TRUE[graph_trial][k],1) for k in range(K)])/np.sum(L)
                    print(f'      Error t: {t_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')')
                    exp_status_file = open(status_file_name,'a')
                    exp_status_file.write(f'      Error t: {t_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')' + '\n')
                    exp_status_file.close()

                    if any(method_iter==np.array([2,4,7])):
                        w_EST[method_iter][num_sig_trial][graph_trial] = w_curr.copy()
                        w_ERROR[method_iter][graph_trial,num_sig_trial] = np.linalg.norm(w_curr-w_TRUE,1)/J
                        print(f'      Error w: {w_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')')
                        exp_status_file = open(status_file_name,'a')
                        exp_status_file.write(f'      Error w: {w_ERROR[method_iter][graph_trial,num_sig_trial]:.4f} ('+method_names[method_iter]+')' + '\n')
                        exp_status_file.close()

                print('')
        #-------------------------

    #-------------------------
    for method_iter in range(num_methods):
        if method_vec[method_iter]:
            print(f'  Mean error s: {np.mean(s_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')')
            exp_status_file = open(status_file_name,'a')
            exp_status_file.write(f'  Mean error s: {np.mean(s_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')' + '\n')
            exp_status_file.close()

            print(f'  Mean error sbar: {np.mean(sbar_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')')
            exp_status_file = open(status_file_name,'a')
            exp_status_file.write(f'  Mean error sbar: {np.mean(sbar_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')' + '\n')
            exp_status_file.close()

            if any(method_iter==np.array([1,2,3,4,6,7])):
                print(f'  Mean error t: {np.mean(t_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')')
                exp_status_file = open(status_file_name,'a')
                exp_status_file.write(f'  Mean error t: {np.mean(t_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')' + '\n')
                exp_status_file.close()

                if any(method_iter==np.array([2,4,7])):
                    print(f'  Mean error w: {np.mean(w_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')')
                    exp_status_file = open(status_file_name,'a')
                    exp_status_file.write(f'  Mean error w: {np.mean(w_ERROR[method_iter][:graph_trial+1,num_sig_trial]):.4f} ('+method_names[method_iter]+')' + '\n')
                    exp_status_file.close()

            print('')
    #-------------------------















# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save simulation (same for all experiments)
print('Saving simulation...')
exp_status_file = open(status_file_name,'a')
exp_status_file.write('Saving simulation...' + '\n')
exp_status_file.close()

s_ERROR_ALL    = np.array([]).reshape(num_graph_trials,-1)
sbar_ERROR_ALL = np.array([]).reshape(num_graph_trials,-1)
t_ERROR_ALL    = np.array([]).reshape(num_graph_trials,-1)
w_ERROR_ALL    = np.array([]).reshape(num_graph_trials,-1)
time_ALL       = np.array([]).reshape(num_graph_trials,-1)
for method_iter in range(num_methods):
    s_ERROR_ALL    = np.concatenate((s_ERROR_ALL,s_ERROR[method_iter]),axis=1)
    sbar_ERROR_ALL = np.concatenate((sbar_ERROR_ALL,sbar_ERROR[method_iter]),axis=1)
    t_ERROR_ALL    = np.concatenate((t_ERROR_ALL,t_ERROR[method_iter]),axis=1)
    w_ERROR_ALL    = np.concatenate((w_ERROR_ALL,w_ERROR[method_iter]),axis=1)
    time_ALL       = np.concatenate((time_ALL,time_method[method_iter]),axis=1)

# sims_params.csv:
#   First row:
#       num_graph_trials: no. trials
#       K: no. graphs
#       G: nodes of discrete graphon
#       N: vector of nodes per graph
#       graphon_fam: Type of graphon used to generate
#   Second row:
#       signal_range: vector of observed signals
#   Third row:
#       Methods used:
#           Column 1: sparse_stat
#           Column 2: stat_mod1
#           Column 3: stat_mod2
#           Column 4: sparse_stat_mod1
#           Column 5: sparse_stat_mod2
#           Column 6: pairdiff_stat
#           Column 7: pairdiff_stat_mod1
#           Column 8: pairdiff_stat_mod2
with open(folder_path+'/'+experiment_name+'/'+'sim_params'+'.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(np.concatenate(([num_graph_trials,K,G],N)).reshape(-1))
    writer.writerow(signal_range.reshape(-1))
    writer.writerow(method_vec.reshape(-1))

# error.csv:
#   Rows:
#       1st num_graph_trials rows: error s
#       2nd num_graph_trials rows: error t
#       3rd num_graph_trials rows: error w
#   Columns:
#       Every len_signal_range columns is new method
#       1st len_signal_range columns: sparse_stat
#       2nd len_signal_range columns: stat_mod1
#       3rd len_signal_range columns: stat_mod2
#       4th len_signal_range columns: sparse_stat_mod1
#       5th len_signal_range columns: sparse_stat_mod2
#       6th len_signal_range columns: pairdiff_stat
#       7th len_signal_range columns: pairdiff_stat_mod1
#       8th len_signal_range columns: pairdiff_stat_mod2
#       -1 fills in methods without result (e.g., sparse_stat has no error t or w)
with open(folder_path+'/'+experiment_name+'/'+'error'+'.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for i in range(num_graph_trials):
        writer.writerow(s_ERROR_ALL[i,:])
    for i in range(num_graph_trials):
        writer.writerow(sbar_ERROR_ALL[i,:])
    for i in range(num_graph_trials):
        writer.writerow(t_ERROR_ALL[i,:])
    for i in range(num_graph_trials):
        writer.writerow(w_ERROR_ALL[i,:])
with open(folder_path+'/'+experiment_name+'/'+'time'+'.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for i in range(num_graph_trials):
        writer.writerow(time_ALL[i,:])


# Save estimates
# est_methodname_#sigs.csv:
#   1st num_graph_trials rows:
#       Each row is concatenation of lower triangle of estimated s
#   2nd num_graph_trials rows (if applicable):
#       Each row is concatenation of lower triangle of estimated t
#   3rd num_graph_trials rows (if applicable):
#       Each row is lower triangle of estimated w
for num_sig_trial in range(len_signal_range):
    num_signals = signal_range[num_sig_trial]
    for method_iter in range(num_methods):
        if method_vec[method_iter]:
            with open(folder_path+'/'+experiment_name+'/'+
                      'est_'+method_names[method_iter]+'_'+str(num_signals)+'sigs'+'.csv','w') as f:
                writer = csv.writer(f,delimiter=',')
                # Save graph estimates
                for graph_trial in range(num_graph_trials):
                    writer.writerow(np.concatenate(s_EST[method_iter][num_sig_trial][graph_trial]))
                for graph_trial in range(num_graph_trials):
                    writer.writerow(np.concatenate(sbar_EST[method_iter][num_sig_trial][graph_trial]))
                if any(method_iter==np.array([1,2,3,4,6,7])):
                    # Save prob. mat. estimates
                    for graph_trial in range(num_graph_trials):
                        writer.writerow(np.concatenate(t_EST[method_iter][num_sig_trial][graph_trial]))
                    if any(method_iter==np.array([2,4,7])):
                        # Save graphon estimates
                        for graph_trial in range(num_graph_trials):
                            writer.writerow(w_EST[method_iter][num_sig_trial][graph_trial])

# truth.csv:
#   1st num_graph_trials rows:
#       Row is concatenation of lower triangle of true s
#   2nd num_graph_trials rows:
#       Row is concatenation of lower triangle of true t
#   Last row:
#       Row is lower triangle of true w
with open(folder_path+'/'+experiment_name+'/'+
          'truth'+'.csv','w') as f:
    writer = csv.writer(f,delimiter=',')
    for graph_trial in range(num_graph_trials):
        writer.writerow(np.concatenate(s_TRUE[graph_trial]))
    for graph_trial in range(num_graph_trials):
        writer.writerow(np.concatenate(t_TRUE[graph_trial]))
    writer.writerow(w_TRUE)

