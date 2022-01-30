#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:35:06 2021

@author: Juan M. Vargas and Mohamed A. Bahloul
"""

#%% Libraries 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RationalQuadratic, RBF, Matern,DotProduct, WhiteKernel,ConstantKernel
from scipy.integrate import simps
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm

from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestRegressor

from scipy.integrate import simps
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import diags
from sklearn.linear_model import LinearRegression
import random
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from numpy import sum,isrealobj,sqrt
from numpy.random import rand

import os

#%% Functions

def arun(s,SNRdB,L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
"""
    gamma = 10**(SNRdB/10) #SNR to linear scale
   
    P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    N0=P/gamma # Find the noise spectral density
    n = sqrt(N0/2)*rand(s.shape[0],s.shape[1]) # computed noise
    r = s + n # received signal
    return r

    # Importing the libraries 
def RMSE(y_pred_lr,y):
    
    """
    RMSE
    Calculate the RMSE between the real value (golden true) and the value predicted by the algorithm.
    Parameters:
        y_pred_lr: vector of values predicted by the model
        y : the real values to be predicted
       
    Returns:
        the values of the RMSE
    """

    return np.sqrt(np.sum(((y_pred_lr-y)**2/len(y_pred_lr))))


def bland_altman_plot(m1, m2,
                      sd_limit=1.96,
                      ax=None,
                      scatter_kwds=None,
                      mean_line_kwds=None,
                      limit_lines_kwds=None):
    """    
    Parameters
    ----------
    m1, m2: pandas Series or array-like
    m1: actually value
    sd_limit : float, default 1.96
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted will be
                       md - sd_limit * sd, md + sd_limit * sd
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences.
        If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
        defaults to 3 standard deviatons on either side of the mean.
    ax: matplotlib.axis, optional
        matplotlib axis object to plot on.
    scatter_kwargs: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
   Returns
    -------
    ax: matplotlib Axis object
    """
    # Importing the libraries 
    import matplotlib.pyplot as plt 
    import numpy as np
    
    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    diffs = m1 - m2
    mean = (m1 + m2) / 2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    if ax is None:
        ax = plt.gca()

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(mean, diffs, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate('mean diff:\n{:.2}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=18,
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 2)),
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=18,
                    xycoords='axes fraction')
        ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 2)),
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=18,
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    # ax.set_ylabel('Difference', fontsize=15)
    # ax.set_xlabel('Actual Value', fontsize=15)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    return ax




def delta(n, fex, feh):
    """
    This function create the Delta matrix used to compute the 1D-SCSA eigenvalues

    """
    ex = np.kron([x for x in range(n-1, 0, -1)], np.ones((n,1)))
    if (n%2) == 0:
        dx = -np.pi**2/(3*feh**2)-(1/6)*np.ones((n,1))
        test_bx = -(-1)**ex*(0.5)/(np.sin(ex*feh*0.5)**2)
        test_tx =  -(-1)**(-ex)*(0.5)/(np.sin((-ex)*feh*0.5)**2)
    else:
        dx = -np.pi**2/(3*feh**2)-(1/12)*np.ones((n,1))
        test_bx = -0.5*((-1)**ex)*np.tan(ex*feh*0.5)**-1/(np.sin(ex*feh*0.5))
        test_tx = -0.5*((-1)**(-ex))*np.tan((-ex)*feh*0.5)**-1/(np.sin((-ex)*feh*0.5))
    
    rng = [x for x in range(-n+1, 1, 1)] + [y for y in range(n-1, 0, -1)]    
    Ex = diags(np.concatenate((test_bx, dx, test_tx), axis = 1).T, np.array(rng), shape = (n, n)).toarray()
    Dx=(feh/fex)**2*Ex
    return Dx



def scsa(y, D,h):
    """
    Compute the 1D-SCSA

    Parameters
    ----------
    y : Is a vector corresponding to a real positive signal
    D : Is the Delta matrix
    h : Is the semiclassical constant

    Returns
    -------
    yscsa : Signal recostructed by SCSA
    kappa : Vector of the eigenvalues 
    Nh : Number of eigenvalues
    psinnor : Matrix with eigenfunctions

    """
    y_max=np.max(y)
    fe = 1
    Y=np.diagflat(y)
    gm = 0.5
    Lcl = (1/(2*(np.pi)**0.5))*(gamma(gm+1)/gamma(gm+(3/2)))
    SC = -h*h*D-Y
    lamda, psi = np.linalg.eigh(SC)
    temp = np.diag(lamda)
    ind = np.where(temp < 0)
    temp = temp[temp < 0]
    kappa = np.diag((-temp)**gm)
    Nh = kappa.shape[0]
    psin = psi[:, ind[0]]
    I = simps(psin**2, dx = fe, axis = 0)
    psinnor = psin/I**0.5   
    yscsa =((h/Lcl)*np.sum((psinnor**2)@kappa,1))**(2/(1+2*gm)) 

    if y.shape != yscsa.shape: yscsa = yscsa.T
    return yscsa, kappa, Nh, psinnor



def scsa_hopt(y, D,v):
    """
    Implement the C-SCSA algorithm to find the optimal h and filter the signal.

    Parameters
    ----------
    y : Input real positive signal
    D : Delta matrix
    v : Value for the curvature weight

    Returns
    -------
    yscsa_op : Signal recostructed
    kappa_op : Vector of eigenvalues
    Nh_op : Number of eigenvalues
    psinnor_op : Matrix of eigenfunctions
    h_op : Optimal value of h

    """
    y=y.real
    y_max=np.max(y)
    if norm==0:
        hh=np.arange(y_max/100,y_max,1)
    hh=np.arange(1,30,1)
    hh= sorted(hh, reverse=True) 
    Cost_function=[]
    for i in range(len(hh)):
      h=hh[i]
      
      # Calculate the SCSA 
      yscsa, kappa, Nh, psinnor=scsa(y, D,h)

      # Cost function J

      ## Accuracy penalty
      y_dif=(y-yscsa)**2
      c_acc=np.sum(y_dif)

      ## Curvature penalty
      y_p1=np.gradient(yscsa)
      y_p2=np.gradient(yscsa)
      kc=np.absolute(y_p2)/(1+y_p1**2)**1.5
      c_cuv=np.sum(kc)
      miu=(1/np.sum(c_cuv))*10**v
      
      ## Compute the cost function
      J=c_acc+(miu*c_cuv)   
      Cost_function.append(J)

    b=np.min(Cost_function)
    a=Cost_function.index(b)
    h_pos=np.min(a)
    
  
    #print(hh.shape)
    h_op=hh[h_pos]
    #print(h_op)
    yscsa_op, kappa_op, Nh_op, psinnor_op=scsa(y, D,h_op)
    
    return yscsa_op, kappa_op, Nh_op, psinnor_op,h_op

def SCSA_FE(signals,v,norm,SNR):
    
    # Add Noise


        
    fe=1
    fe_vec=[]  # Empty list that contain 
    
    h_v=[]
    ne=[]
    
    num_plot=5 # Number of plots we want to show 
    plots=[random.randint(0,signals.shape[0]) for i in range(num_plot)] #Random number to select the plots
    plots=[0,1,2,3,4]
    print('Starting features extraction.....')
    # Calculate First,Second and Thrid derivative
    for s in range(signals.shape[0]):
        #print(sig)
        exists = s in plots    
        sig_ori=signals[s]
    ##########################################################################################   

        sig_no=arun(sig_ori,SNR)
        
        if norm==1: 
            sig=sig_no/np.max(sig_no)#Constants
        else:
            sig=sig_no
    
    ###############################################################################################    
        
        # compute the SCSA 
        sig=np.transpose(sig)
        t=sig.shape[1]
        tri=2*np.pi/t
        D= delta(t,fe,tri)
        yscsa, kappa, Nh, psinnor,h= scsa_hopt(sig, D,v)
        h_v.append(h)
        ne.append(len(kappa))
     
    
        
    
    ######################################################################################################    
        
        
        # Features estimation
        
        #1. Features for the First derivate of the PPG
        
        kappa_o=np.diag(kappa);
        
        # a. SCSA Invariants 
        INV1=4*h*np.sum(kappa_o);
        INV2=((16*h)/3) *np.sum(kappa_o**3);
        INV3=((256*h)/7) *np.sum(kappa_o**7);
        
        # b. Stadistics features for eigenvalues (Kappas)
        kpmean=np.mean(kappa_o);
        kpmed=np.median(kappa_o);
        kpstd=np.std(kappa_o);
        
    
        # c. First tree eigenvalues
        f_eigenvalue=kappa_o[0];
        s_eigenvalue=kappa_o[1];
        t_eigenvalue=kappa_o[2];
        
        
        # d. First tree squared eigenvalues
          
        f_eigenvalue_s=(kappa_o[0])**2;
        s_eigenvalue_s=(kappa_o[1])**2;
        t_eigenvalue_s=(kappa_o[2])**2;
        
        # e. Number of eigenvalues
        N_eig=Nh
        
        # f. Ratios between eigenvalues and h
        Kmer=np.mean(kappa_o[0])/h
        Kmedr=np.median(kappa_o[0])/h
          
    
    
    #########################################################################################################
        #Compute feature matrix

          
        fe_vec.append([INV1,INV2,INV3,kpmean,kpmed,kpstd,f_eigenvalue,s_eigenvalue,t_eigenvalue,f_eigenvalue_s,s_eigenvalue_s,t_eigenvalue_s, N_eig,Kmer,Kmedr])
        
    

        
        #all_fe=np.concatenate((fe_vec, medical_features), axis=1)
    pd_fe=pd.DataFrame(fe_vec,columns=['INV1','INV2','INV3','kpmean','kpmed','kpstd','f_eigenvalue','s_eigenvalue','t_eigenvalue','f_eigenvalue_s','s_eigenvalue_s','t_eigenvalue_s', 'N_eig','Kmer','Kmedr'])
    pd_fe_f=pd_fe.dropna(how='any')    
    pd_fe_f.to_csv('./Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)+'/' +'1D-SCSA_features.csv', index = False)
 
    
    hne_df=np.concatenate((np.asarray(h_v).reshape(-1,1),np.asarray(ne).reshape(-1,1)),axis=1)
    h_df=pd.DataFrame(hne_df,columns=['h','ne'])
    h_df.to_csv('./Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)+'/' +'1D-SCSA_optimal_h.csv', index = False)
    
    print('Ending features extraction.....')
    return pd_fe_f



   

#%% Load signals

#Data characteristic definition

type_sig='Digital' # Location of mesure: Brachial, Radial or Digital (Finger)
type_wav='PPG' # Typer of signal: BP or PPG
SNR=20  # Level of noise (dB)
norm=1 # 0 is for no normalize data and 1 for normalize data
h=0.1  # Semi-classical constant used      # only want features from original signal
gam=3  # gamma value for scsa
med_f='no' # used clinical rutine data
       


print('RUNNING FOR  SIGNAL '+type_sig+'_'+type_wav+' h='+str(h)+' norm='+str(norm)+' SNR='+str(SNR))

# Create folder
path='./Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)


if os.path.isdir(path)==False:
     os.makedirs(path)


# Load the Digital signal and target
mat = scipy.io.loadmat('./Data/'+type_sig+'_'+type_wav+'.mat')
signals=mat['ppg_dig']    


pwv= scipy.io.loadmat('./Data/PWVcf.mat')
PWV_cf=np.transpose(pwv['haemods'].astype('float64'))

signals=signals[0,0:signals.shape[1]]

signals_p=signals

#%%  1D-SCSA feature extraction and feature selection

v1=1

features=SCSA_FE(signals,v1,norm,SNR)
fe=features.values
corr=[]
for i in range(features.shape[1]):
    f_f=scipy.stats.pearsonr(fe[:,i].reshape(-1,), PWV_cf.reshape(-1,))
    corr.append(f_f[0])

corr_v=np.array(corr)

corr_v[np.abs(corr_v)<=0.5]=0
boolArr = (corr_v != 0)
result = np.where(boolArr)
ff=fe[result[0]]


if med_f=='yes':
    medical_data=pd.read_csv('./Data/pwdb_haemod_params.csv')
    features_m=[' age [years]',' HR [bpm]',' SBP_b [mmHg]',' DBP_b [mmHg]',' MBP_b [mmHg]',' PP_b [mmHg]',' PWV_cf [m/s]']
    medical_features=medical_data[features_m]
    features=np.concatenate((features,medical_features),axis=1)


X_train,X_test,PWV_cf_train,PWV_cf_test=train_test_split(features,PWV_cf, test_size=0.3, random_state=31)
 




sc = StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)


X_test=sc.transform(X_test)

y_test=PWV_cf_test.reshape(-1,)












#%% Linear regression Training and testing

print('Strart classification using LR')

# Set hyper-parameter space
hyper_params = [{'fit_intercept':[True,False]}]

# Create linear regression model 
lm = LinearRegression()
# Create RandomSearchCV() with 5-fold cross-validation
model_cv = RandomizedSearchCV(estimator = lm,param_distributions=hyper_params,n_iter=5,cv = 5,random_state=42)  

# Fit the model
model_cv.fit(X_train,PWV_cf_train.reshape(-1,))

# Test model
y_pred_lr =model_cv.predict(X_test)

#%% Save results and graphs

RMSE_LR=RMSE(y_pred_lr,y_test)
err_LR=(RMSE_LR/np.mean(y_test))*100
vee=[[RMSE_LR,err_LR]]
LR_result= pd.DataFrame(vee, columns = ['RMSE','per_err'])
LR_result.to_csv('./Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)+'/' +'LR_metrics.csv')


# Plot fig
# blandAltman plot
fig1 = plt.figure(figsize = (8,8))
ax1 = fig1.add_subplot(1,1,1)
ax1 = bland_altman_plot(y_test, y_pred_lr)
plt.ylim(-5,6)
ax1.tick_params(axis='x',labelsize = 20)
ax1.tick_params(axis='y',labelsize = 20)
plt.xticks(np.arange(6, 19, 6))
plt.yticks(np.arange(-4, 5, 4))
filename1 = './Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)+'/' +'LR_Bland_Altman.png'
fig1.savefig(filename1)

# Estimated vs measured
m, b = np.polyfit(y_pred_lr, y_test,1)
X = sm.add_constant(y_pred_lr)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared
fig2 = plt.figure(figsize = (8,8))
ax2 = fig2.add_subplot(1,1,1)
plt.plot(y_pred_lr, y_test, 'k.', markersize = 10)
ax2.plot(y_pred_lr, m*y_pred_lr +b, 'r', label = 'y = {:.2f}x+{:.2f}'.format(m, b))
plt.xlim(0,18)
plt.ylim(0,25)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.xticks(np.arange(1, 16, 6))
plt.yticks(np.arange(1, 22, 6))
plt.legend(fontsize=18,loc=2)
ax2.text(4, 21, 'r$^2$ = {:.2f}'.format(r_squared), fontsize=18)
ax2.text(4, 19, 'p < 0.0001', fontsize=18)
filename2 = './Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)+'/' +'LR_est_vs_med.png'
fig2.savefig(filename2)

# save target_test and target_pred
savedata = [y_test, y_pred_lr]
df_savedata = pd.DataFrame(savedata)
df_savedata.to_csv('./Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)+'/' +'y_no_pred_LR.csv')

hp_LR=pd.DataFrame(model_cv.best_params_.items())
hp_LR.to_csv('./Data/1D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gam)+'_SNR='+str(SNR)+'/' +'LR_hyperparameters.csv')


print('Ending classification using LR')
