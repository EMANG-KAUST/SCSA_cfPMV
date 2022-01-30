#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:35:06 2021

@authors: Juan Manuel Vargas and Mohamed A. Bahloul 
"""
#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

#%% Functions
    # Importing the libraries
def RMSE(y_pred_lr,y):
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





#%% Load 2D-SCSA features extracted

#Data characteristic definition

type_sig='Digital' # Location of mesure: Brachial, Radial or Digital (Finger)
type_wav='BP' # Typer of signal: BP or PPG
SNR=500  # Level of noise (dB)
norm=1 # 0 is for no normalize data and 1 for normalize data
h=0.1  # Semi-classical constant used
gamma=3       # only want features from original signal
med_f='no' # used clinical rutine data


print('RUNNING FOR  SIGNAL '+type_sig+'_'+type_wav+' h='+str(h)+' norm='+str(norm)+' SNR='+str(SNR))

features=pd.read_csv('./Data/2D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gamma)+'_SNR='+str(SNR)+'/features_final.csv',header=None)
y=pd.read_csv('./Data/PWV_cf.csv',header=None)

X_f=features.values
PWV_cf=np.transpose(y.values)        
if med_f=='yes':        
    medical_data=pd.read_csv('./Data/pwdb_haemod_params.csv')    
    features_m=[' age [years]',' HR [bpm]',' SBP_b [mmHg]',' DBP_b [mmHg]',' MBP_b [mmHg]',' PP_b [mmHg]',' PWV_cf [m/s]']
    medical_features=medical_data[features_m]
    X_f=np.concatenate((X_f,medical_features),axis=1)


#%% Data pre-proccesing

X_train,X_test,PWV_cf_train,PWV_cf_test=train_test_split(X_f,PWV_cf, test_size=0.3, random_state=31)


 
# Data standarization

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
LR_result.to_csv('./Data/2D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gamma)+'_SNR='+str(SNR)+'/' +'LR_metrics.csv')


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
filename1 = './Data/2D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gamma)+'_SNR='+str(SNR)+'/' +'LR_Bland_Altman.png'
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
filename2 = './Data/2D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gamma)+'_SNR='+str(SNR)+'/' +'LR_est_vs_med.png'
fig2.savefig(filename2)

# save target_test and target_pred
savedata = [y_test, y_pred_lr]
df_savedata = pd.DataFrame(savedata)
df_savedata.to_csv('./Data/2D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gamma)+'_SNR='+str(SNR)+'/' +'y_no_pred_LR.csv')

hp_LR=pd.DataFrame(model_cv.best_params_.items())
hp_LR.to_csv('./Data/2D-SCSA' +type_wav+'_'+type_sig+'_h='+str(h)+'_gamma='+str(gamma)+'_SNR='+str(SNR)+'/' +'LR_hyperparameters.csv')


print('Ending classification using LR')
