"""
lc_analysis.py

Created 08/2022

Original Author: Ryne Dingler

Some functions useful for operations are obtained from 
Connolly 2015: http://arxiv.org/abs/1503.06676

Python version of the light curve variability statistic algorithm from 
Dingler & Smith 2023.


requires:
    os, sys, numpy, scipy, pandas, csv, matplotlib

"""


import os
import sys
import math
import pandas as pd
import numpy as np
import csv
import scipy.optimize as op
import scipy.special as sp
from scipy.optimize import curve_fit
from scipy import stats as st
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')

###########################################################################################################
###########################################################################################################
'''
Definition of various functions which are used to fit distribution models and linear fits 
'''

def linear(x,m,b):
    return(m*x+b)

def gaussian(x, A, μ, σ):
    return(A * np.exp(-(x - μ)**2 / (2*σ**2)))

def lognormal(x, A, μ, σ):
    return(1/x * gaussian(np.log(x), A, μ, σ))

def bimodal(x, p, A, μ1, σ1, B, μ2, σ2):
    ϕ = 0.5 + np.arctan(p)/np.pi
    return(ϕ * gaussian(x, A, μ1, σ1) + (1-ϕ) * gaussian(x, B, μ2,σ2))


###########################################################################################################
###########################################################################################################
'''
Definition of various functions regarding rounding 
'''

def roundedfractionalrange(arr,fraction = 0.5):
    """
    Returns a value for fractional length of an array.
    """
    return(int(fraction*len(arr) + 0.5))

def round_up_decimals(num:float, dec:int):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    round_good = False
    count = 0
    while round_good == False:

        if not isinstance(dec, int) or dec < 0:
            raise TypeError("decimal places must be a positive integer")
            if count != 0:
                dec = int(input("round to how many decimal places? "))
        elif dec == 0:
            return(np.ceil(num))
            round_good = True
        else:
            round_good = True
        count += 1   

    scale = 10**dec
    return(np.ceil(num * scale) / scale)

###########################################################################################################
###########################################################################################################
def Min_PDF(params,hist,model,force_scipy=True):
    '''
    PDF chi squared function allowing the fitting of a mixture distribution
    using a log normal distribution and a gamma distribution
    to a histogram of a data set.
    
    inputs:
        params (array)   - function variables - kappa, theta, lnmu,lnsig,weight
        hist (array)     - histogram of data set (using numpy.hist)
        force_scipy (bool,optional) - force the function to assume a scipy model
    outputs:
        chi (float) - chi squared
        
    Ref: Connolly 2015
    '''

    mids = (hist[1][:-1]+hist[1][1:])/2.0

    try:
        if model.__name__ == 'Mixture_Dist':
            model = model.Value
            m = model(mids,params)
        elif model.__module__ == 'scipy.stats.distributions' or \
            model.__module__ == 'scipy.stats._continuous_distns' or \
                force_scipy == True:
            m = model.pdf    
        else:    
            m = model(mids,*params)
    except AttributeError:
        m = model(mids,*params)
    
    chi = (hist[0] - m)**2.0
    
    return(np.sum(chi))

def OptBins(data,maxM=100):
    '''
     Python version of the 'optBINS' algorithm by Knuth et al. (2006) - finds 
     the optimal number of bins for a one-dimensional data set using the 
     posterior probability for the number of bins. WARNING sometimes doesn't
     seem to produce a high enough number by some way...
    
     inputs:
         data (array)           - The data set to be binned
         maxM (int, optional)   - The maximum number of bins to consider
         
     outputs:
        maximum (int)           - The optimum number of bins
    
     Ref: Connolly 2015, K.H. Knuth. 2012. Optimal data-based binning for histograms
     and histogram-based probability density models, Entropy.
    '''
    
    N = len(data)
    
    # loop through the different numbers of bins
    # and compute the posterior probability for each.
    
    logp = np.zeros(maxM)
    
    for M in range(1,maxM+1):
        n = np.histogram(data,bins=M)[0] # Bin the data (equal width bins)
        
        # calculate posterior probability
        part1 = N * np.log(M) + sp.gammaln(M/2.0)
        part2 = - M * sp.gammaln(0.5)  - sp.gammaln(N + M/2.0)
        part3 = np.sum(sp.gammaln(n+0.5))
        logp[M-1] = part1 + part2 + part3 # add to array of posteriors

    maximum = np.argmax(logp) + 1 # find bin number of maximum probability
    return(maximum + 10)

###########################################################################################################
###########################################################################################################

def check_col_label(rootdir,bin1,bin2,sample = "", raw = False):
    '''
    Make sure data file to save results exists and has proper headers
    
    inputs:
        rootdir (directory)       - The directory in which figures will be saved
        bin1 (string)             - string which specified how many hours for one binning schema
        bin2 (string)             - string which specified how many hours for one binning schema
        sample (string)           - user-specified sample selection
        raw (boolean)             - whether or not raw light curve is available
         
   '''
    
    col_title = False
    try:
        if sample =="analysis":
            lc_data = open(os.path.join(rootdir,'LC_variability_analysis.csv') , "r+")
        if sample =="removed":
            lc_data = open(os.path.join(rootdir,'LC_variability_rmv.csv') , "r+")

    
    except FileNotFoundError:
        if sample =="analysis":
            lc_data = open(os.path.join(rootdir,'LC_variability_analysis.csv') , "w")
            print("\nCreating file 'LC_variability_analysis.csv'\n")
        if sample =="removed":
            lc_data = open(os.path.join(rootdir,'LC_variability_rmv.csv') , "w")
        print("\nCreating file 'LC_variability_rmv.csv'\n")
        
        
        if raw == True:
            header = ['Target','<F>(raw)','σ_F(raw)','<ΔF>(raw)', 'σ_ΔF(raw)',\
            '<F>(unbinned)', '<F>('+bin1+'hr)','<F>('+bin2+'hr)',\
                'σ_F(unbinned)','σ_F('+bin1+'hr)', 'σ_F('+bin2+'hr)',\
                    '<ΔF>(unbinned)','<ΔF>('+bin1+'hr)','<ΔF>('+bin2+'hr)',\
                        'σ_ΔF(unbinned)', 'σ_ΔF('+bin1+'hr)','σ_ΔF('+bin2+'hr)',\
                            'σ^2_NXS(unbinned)','err(σ^2_NXS)(unbinned)','σ^2_NXS('+bin1+'hr)','err(σ^2_NXS)('+bin1+'hr)','σ^2_NXS('+bin2+'hr)','err(σ^2_NXS)('+bin2+'hr)',\
                                'F_var(unbinned)','err(F_var)(unbinned)','F_var('+bin1+'hr)','err(F_var)('+bin1+'hr)','F_var('+bin2+'hr)','err(F_var)('+bin2+'hr)',\
                                    'τ(inc,days)', 'Δt(inc,days)', 'τ(dec,days)', 'Δt(dec,days)','χ^2','dof','χ^2/dof']
        
        if raw == False:
            header = ['Target','<F>(unbinned)', '<F>('+bin1+'hr)','<F>('+bin2+'hr)',\
            'σ_F(unbinned)','σ_F('+bin1+'hr)', 'σ_F('+bin2+'hr)',\
                '<ΔF>(unbinned)','<ΔF>('+bin1+'hr)','<ΔF>('+bin2+'hr)',\
                    'σ_ΔF(unbinned)', 'σ_ΔF('+bin1+'hr)','σ_ΔF('+bin2+'hr)',\
                        '<σ^2_XS>(unbinned)','<σ^2_XS>('+bin1+'hr)','<σ^2_XS>('+bin2+'hr)',\
                            'F_var(unbinned)','<F_var>('+bin1+'hr)','<F_var>('+bin2+'hr)',\
                                'τ(inc,days)', 'Δt(inc,days)', 'τ(dec,days)', 'Δt(dec,days)','χ^2','dof','χ^2/dof']

        
        writer = csv.writer(lc_data)
        writer.writerow(header)

    lc_data.close()


    return()
###########################################################################################################
###########################################################################################################
'''
Definition of various functions regarding statistical tests 
'''

def chisquare_per_dof(observed,expected,error):
    '''
    Chi-squared per degree of freedom test
    
    inputs:
        observed (array or float)    - observed test values
        expected (array or float)    - theoretical values
        error (array or float)       - observed error on test values
    outputs:
        test_statistic (float)       - chi-square statistic 
        dof (float)                  - length of array
        test_statistic/dof (float)   - chi-square per dof
    '''
    
    index = np.array(list(np.where(error != 0)[0]), dtype = 'int')
    test_statistic = []
    for idx in index:
        test_statistic.append(np.subtract(observed,expected)[idx]**2 / error[idx]**2)
        
    dof=len(test_statistic)-1
    test_statistic = np.nansum(test_statistic)

    return(test_statistic, dof, test_statistic/dof)


def shortest_timescale(rootdir, target, uninterp_df, sectbysect = False, sectors = ''): 
    '''
    Find shortest timescale for exponential rise and decay for significant flux change
    with decreasing iterative kernel smoothing. Smoothing aid in the reduction of single
    cadence flares; however, may eliminate significant flux change, hence the decreasing iterations.
    
    inputs:
         rootdir (directory)          - The directory in which lcs are stored
         target (string)              - Name of desired object to analyze
         uninterp_df (DataFrame)      - Number of lcs (default = 501)
         redshift (float)             - Redshift of AGN
         sectbysect (boolean)         - sector by sector analysis or not (default is False)
         secors (string)              - current sectors under analysis if sectbysect is True
         
     outputs:
        tau_dec_min (float)           - shortest timescale of exponential decay
        delt_dec_min (float)          - assosiacted time seperation for timescale 
                                            of exponential decay
        tau_inc_min (float)           - shortest timescale of exponential growth
        delt_inc_min (float)          - assosiacted time seperation for timescale 
                                            of exponential growth
    
     Ref:  Chaterjee et al 2021
    '''
    
    nonzeroflux = np.nonzero(np.array(uninterp_df[1]))[0]
    time = np.array(uninterp_df[0][nonzeroflux])
    unfiltered_flux = np.array(uninterp_df[1][nonzeroflux])
    flux = []
    err = np.array(uninterp_df[2][nonzeroflux])

    tau_dec_min = np.inf
    tau_inc_min = np.inf
    
    delt_dec_min = 0.
    delt_inc_min = 0.
    
    t1_dec_idx = 0
    t2_dec = 0
    t1_inc_idx = 0
    t2_inc = 0
    
    FWHM = 2

    while tau_dec_min == np.inf or tau_inc_min == np.inf:
        print("\nSmoothing light curve with FWHM = %i"%FWHM)
        flux = gaussian_filter(unfiltered_flux,sigma=FWHM)
        for i in range(0,len(time)-1):
            tmp_time = time[i]
            tmp_flux = flux[i]
            tmp_err = err[i]

            if i%500 == 0.:
                print("Comparing point %i"%i)

            delt = np.subtract(time[i+1:],tmp_time)
            delf = np.subtract(flux[i+1:],tmp_flux)

            denom = np.log(np.divide(flux[i+1:],tmp_flux))
            tau = np.log(2)*np.abs(np.divide(delt,denom))

            stats = np.column_stack((delt,delf,tau))

            delt_dec = [x for x,y,z in stats if x != 0. and y < -3.*tmp_err and np.isfinite(z)]
            tau_dec = [z for x,y,z in stats if x != 0. and y < -3.*tmp_err and np.isfinite(z)]
            delt_inc = [x for x,y,z in stats if x != 0. and y > 3.*tmp_err and np.isfinite(z)]
            tau_inc = [z for x,y,z in stats if x != 0. and y > 3.*tmp_err and np.isfinite(z)]


            if len(tau_dec) != 0:
                if np.min(tau_dec) < tau_dec_min:
                    tau_dec_min = np.min(tau_dec)
                    delt_dec_min = delt_dec[np.argmin(tau_dec)]
                    t1_dec_idx = i
                    t2_dec = time[t1_dec_idx] + delt_dec_min 

            if len(tau_inc) != 0:
                if np.min(tau_inc) < tau_inc_min:
                    tau_inc_min = np.min(tau_inc)
                    delt_inc_min = delt_inc[np.argmin(tau_inc)]
                    t1_inc_idx = i
                    t2_inc = time[t1_inc_idx] + delt_inc_min

        if tau_inc_min != np.inf and tau_dec_min != np.inf:
            break
        elif FWHM == 0:
            if tau_inc_min == np.inf or tau_dec_min == np.inf:
                break
        FWHM-=1
                    
    plt.figure(figsize=(25,6))
    plt.title(target+": shortest timescales of exponential growth and decay")
    plt.errorbar(time,unfiltered_flux,yerr=err, color = 'k', alpha = 0.75, linestyle = 'None')
    plt.vlines((time[t1_dec_idx],t2_dec), np.min(flux)-6, np.max(flux)+6,color = 'r', label = r"$τ_{dec}$ = %.3f"%tau_dec_min)
    plt.vlines((time[t1_inc_idx],t2_inc), np.min(flux)-6, np.max(flux)+6,color = 'b', label = r"$τ_{inc}$ = %.3f"%tau_inc_min)
    if FWHM == 0:
        plt.plot(time,flux, label = 'Running average')
    else:
        plt.plot(time,flux, label = r'Smoothed lc $\sigma$=%i'%FWHM)
    plt.ylim(np.min(flux)-5, np.max(flux)+5)
    plt.legend(loc = 'best')
    plt.tight_layout()
    if sectbysect == True:
        plt.savefig(rootdir+target+'/'+target+'_sectors'+sectors+'_shortest_timescales_smoothedlc.png')
    else:
        plt.savefig(rootdir+target+'/'+target+'_shortest_timescales_smoothedlc.png')
#         plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    plt.title(target+": shortest timescale of exponential decay")
    plt.errorbar(time,unfiltered_flux,yerr=err, color = 'k', alpha = 0.75, linestyle = 'None')
    plt.vlines((time[t1_dec_idx],t2_dec), np.min(flux)-6, np.max(flux)+6,color = 'r', label = r"$τ_{dec}$ = %.3f"%tau_dec_min)
    if FWHM == 0:
        plt.plot(time,flux, label = 'Running average')
    else:
        plt.plot(time,flux, label = r'Smoothed lc $\sigma$=%i'%FWHM)
    plt.xlim(time[t1_dec_idx]-0.5,t2_dec+0.5)
    plt.ylim(np.min(flux)-5, np.max(flux)+5)
    plt.legend(loc = 'best')
    plt.tight_layout()
    if sectbysect == True:
        plt.savefig(rootdir+target+'/'+target+'_sectors'+sectors+'_shortest_decaytimescale_smoothedlc.png')
    else:
        plt.savefig(rootdir+target+'/'+target+'_shortest_decaytimescale_smoothedlc.png')
#         plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    plt.title(target+": shortest timescale of exponential growth")
    plt.errorbar(time,unfiltered_flux,yerr=err, color = 'k', alpha = 0.75, linestyle = 'None')
    plt.vlines((time[t1_inc_idx],t2_inc), np.min(flux)-6, np.max(flux)+6,color = 'b', label = r"$τ_{inc}$ = %.3f"%tau_inc_min)
    if FWHM == 0:
        plt.plot(time,flux, label = 'Running average')
    else:
        plt.plot(time,flux, label = r'Smoothed lc $\sigma$=%i'%FWHM)
    plt.xlim(time[t1_inc_idx]-0.5,t2_inc+0.5)
    plt.ylim(np.min(flux)-5, np.max(flux)+5)
    plt.legend(loc = 'best')
    plt.tight_layout()
    if sectbysect == True:
        plt.savefig(rootdir+target+'/'+target+'_sectors'+sectors+'_shortest_growthtimescale_smoothedlc.png')
    else:
        plt.savefig(rootdir+target+'/'+target+'_shortest_growthtimescale_smoothedlc.png')
#         plt.show()
    plt.close()
    
    
    return(tau_dec_min, delt_dec_min, tau_inc_min, delt_inc_min)

###########################################################################################################

def excess_variance(time, flux, err, stdev = 0, len_lc = 0, MSE = 0, total=True, normalize = True):
    '''
    Find sExcess variance and fractional variance for either the entire light curve or within user-
    specified bins
    
    inputs:
         time (array)             - array of time from the light curve (or segment)
         flux (array)             - array of flux from the light curve (or segment)
         err (array)              - array of error from the light curve (or segment)
         stdev (float)            - standard deviation of entire flux sample
         len_lc (float)           - length of the light curve (or segment)
         MSE (float or array)     - mean squared error from of error from the light curve (or segment)
         total (boolean)          - indicator if analyzing the total light curve or analyzing intrabin
         normalize (boolean)      - indicator if excess variance is to be normalized
         
     outputs (total = False):
        Xvar (float)              - Excess Variance
        Fvar (float)              - Fractional variance
        XvarErr (float)           - error on Excess Variance
        FvarErr (float)           - error on Fractional variance
        
    `outputs (total = True, normalize = True):
        NXvar (float)             - Normalized Excess Variance
        Fvar (float)              - Fractional variance
        NXvarErr (float)          - error on Normalized Excess Variance
        FvarErr (float)           - error on Fractional variance
    
     Ref:  Chaterjee et al 2021
    '''
    
    nonzeroflux = np.nonzero(np.array(flux))[0]
    time = np.array(time[nonzeroflux])
    flux = np.array(flux[nonzeroflux])
    err = np.array(err[nonzeroflux])
    
    if total == True:
        lc_stdev = np.nanstd(flux)
        flux_mean = np.nanmean(flux)

        len_lc = len(time)
        MSE = np.nanmean(err**2)
        Xvar = lc_stdev**2 - MSE
        
        if normalize == True:
            NXvar = Xvar/flux_mean**2
            Fvar = np.sqrt(NXvar)
        else:
            NXvar = Xvar
            Fvar = np.sqrt(NXvar/flux_mean**2)
        
        if np.isfinite(Fvar):
            NXvarErr = np.sqrt(np.add( (np.sqrt(2/len_lc) * MSE/(flux_mean**2))**2,\
                                              (np.sqrt(MSE/len_lc) * (2*Fvar/flux_mean))**2))
        else:
            NXvarErr = np.sqrt(2/len_lc) * MSE/(flux_mean**2)
            
        NXvarErr = np.array(NXvarErr)        
        FvarErr = NXvarErr/(2*Fvar)
    
        return(NXvar, Fvar, NXvarErr, FvarErr)
    
    else:
        
        Xvar = np.subtract(stdev**2,MSE)
        Fvar = np.sqrt(np.divide(Xvar,flux**2))
        XvarErr = []
        for i in range(0,len(Xvar)):
            if np.isfinite(Fvar[i]):
                XvarErr.append(np.sqrt( np.add( (np.sqrt(2/len_lc[i]) * MSE[i]/(flux[i]**2))**2,\
                                (np.sqrt(MSE[i]/len_lc[i]) * (2*Fvar[i]/flux[i]))**2)))
            else:
                XvarErr.append(np.sqrt(2/len_lc[i]) * MSE[i]/(flux[i]**2))

        XvarErr = np.array(XvarErr)        
        FvarErr = XvarErr/(2*Fvar)
    
        return(Xvar, Fvar, XvarErr, FvarErr)     
            
###########################################################################################################################################
        
def plot_excess_variance(subdir, df, binneddf, Xvar, XvarErr, Xvar_mean, Fvar, FvarErr, Fvar_mean, bin_time, sectbysect, group):
    
    '''
    Plot the light curves, binned light curves, itrabin excess/fractional variance, ad confidence interval plots for 
    itrabin excess/fractional variance.
    
    inputs:
        subdir (directory)        - The directory in which figures will be saved
        df (DataFrame)            - dataframe containing unbinned light curve data (time,flux,err)
        binneddf (DataFrame)      - dataframe containing binned light curve data (time,flux,err)
        Xvar (float)              - Excess Variance
        XvarErr (float)           - error on Excess Variance
        Xvar_mean (float)         - mean of Excess Variance
        Fvar (float)              - Fractional variance
        FvarErr (float)           - error on Fractional variance
        Fvar_mean (float)         - mean of Fractional variance
        bin_time (string)         - string which specified how many hours are in one bin of binned lc
        sectbysect (boolean)      - sector by sector analysis or not (default is False)
        group (string)            - current sectors under analysis if sectbysect is True        

    '''
        

    fig = plt.figure(figsize=(8,10))
    gs = gridspec.GridSpec(nrows=4, ncols= 1,hspace=0.0)
    fig.tight_layout()

    ax0 = fig.add_subplot(gs[0, :])
    ax0.errorbar(df[0],df[1],yerr=df[2], linestyle ="None", color = 'k')
    ax0.set_ylabel("F")
    ax0.minorticks_on()
    ax0.tick_params(axis='x', which = 'major', top = True, direction= 'in', length = 8)
    ax0.tick_params(axis='x', which = 'minor', top = True, direction= 'in', length = 4)
    ax0.set_title(target+r': $σ_{XS}^{2}$ & $F_{var}$, '+bin_time+'hr bins',fontsize='x-large')


    ax1 = fig.add_subplot(gs[1, :],sharex=ax0)
    ax1.errorbar(binneddf[0],binneddf[1],yerr=np.sqrt(binneddf[2]), linestyle ="None",  color = 'k')
    ax1.set_ylabel(r"$\langle F \rangle$")
    ax1.minorticks_on()
    ax1.tick_params(axis='x', which = 'major', top = True, direction= 'in', length = 8)
    ax1.tick_params(axis='x', which = 'minor', top = True, direction= 'in', length = 4)


    ax2 = fig.add_subplot(gs[2, :],sharex=ax1)
    ax2.errorbar(binneddf[0], Xvar, yerr = binneddf[1]*XvarErr, linestyle ="None", fmt='+', markersize = 4, color = 'k')

    ax2.set_ylabel(r"$σ_{XS}^{2}$")
    ax2.minorticks_on()
    ax2.tick_params(axis='x', which = 'major', top = True, direction= 'in', length = 8)
    ax2.tick_params(axis='x', which = 'minor', top = True, direction= 'in', length = 4)


    ax3 = fig.add_subplot(gs[3, :],sharex=ax2)
    ax3.errorbar(binneddf[0], Fvar, yerr= FvarErr, linestyle ="None", color = 'k')
    ax3.set_ylabel(r"$F_{var}$")
    ax3.minorticks_on()
    ax3.tick_params(axis='x', which = 'major', top = True, direction= 'in', length = 8)
    ax3.tick_params(axis='x', which = 'minor', top = True, direction= 'in', length = 4)
    ax3.set_xlabel('Time (BTJD - days)')

    plt.savefig(subdir+'/'+target+'_excess_rms_variance_'+bin_time+'hr.pdf', format = 'pdf',bbox_inches='tight')
    #             plt.show()
    plt.close()


    try:
        fig = plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(nrows=2, ncols= 1,hspace=0.0)
        fig.tight_layout()

        ax2 = fig.add_subplot(gs[0, :])
        ax2.errorbar(binneddf[0], Xvar, yerr = binneddf[1]*XvarErr, linestyle ="None", fmt='+', markersize = 4, color = 'k')
        ax2.axhline(Xvar_mean, color = 'k', linestyle = 'solid', label = r'$\langle σ_{XS}^{2} \rangle$ = %.2e'%Xvar_mean)

        # create custom discrete random variable from data set

        try:
            try:
                unique, counts = np.unique(Xvar, return_counts=True)
                pk = [x/len(Xvar) for x in counts]
                CI_Xvar = st.rv_discrete(a = -np.inf,values=(unique, pk))  
            except:
                CI_Xvar = st.rv_discrete(a = -np.inf,values=(Xvar, np.array(len(Xvar)*[1/len(Xvar)])))                       
            ax2.fill_between([binneddf[0].min()-1, binneddf[0].max()+1], CI_Xvar.interval(0.68)[0], CI_Xvar.interval(0.68)[1], color = 'g', alpha = 0.5)
            ax2.fill_between([binneddf[0].min()-1, binneddf[0].max()+1], CI_Xvar.interval(0.95)[0], CI_Xvar.interval(0.95)[1], color = 'g', alpha = 0.5)

        except:
            CI68_Xvar = st.norm.interval(alpha=0.68, loc= Xvar_mean, scale=np.std(Xvar))
            CI95_Xvar = st.norm.interval(alpha=0.95, loc= Xvar_mean, scale=np.std(Xvar))
            ax2.fill_between([binneddf[0].min()-1, binneddf[0].max()+1], CI68_Xvar[0], CI68_Xvar[1], color = 'g', alpha = 0.5)
            ax2.fill_between([binneddf[0].min()-1, binneddf[0].max()+1], CI95_Xvar[0], CI95_Xvar[1], color = 'g', alpha = 0.5)


        ax2.set_ylabel(r"$σ_{XS}^{2}$")
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which = 'major', top = True, direction= 'in', length = 8)
        ax2.tick_params(axis='x', which = 'minor', top = True, direction= 'in', length = 4)
        ax2.set_title(target+r': $σ_{XS}^{2}$ & $F_{var}$, '+bin_time+'hr bins',fontsize='x-large')
        ax2.legend(loc = 'upper right')

        ax3 = fig.add_subplot(gs[1, :],sharex=ax2)
        ax3.errorbar(binneddf[0], Fvar, yerr= FvarErr, linestyle ="None", color = 'k')
        ax3.axhline(Fvar_mean, color = 'k', linestyle = 'solid', label = r'$\langle F_{var} \rangle$ = %.2e'%Fvar_mean)

        # create custom discrete random variable from data set
        Fvar_finite = [x for x in Fvar if np.isfinite(x)]
        try:
            try:
                unique, counts = np.unique(Fvar_finite , return_counts=True)
                pk = [x/len(Fvar_finite) for x in counts]
                CI_Fvar = st.rv_discrete(a = -np.inf,values=(unique, pk))  
            except:
                CI_Fvar = st.rv_discrete(a = -np.inf,values=(Fvar_finite,np.array(len(Fvar_finite)*[1/len(Fvar_finite)])))
            ax3.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI_Fvar.interval(0.68)[0], CI_Fvar.interval(0.68)[1], color = 'g', alpha = 0.5)
            ax3.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI_Fvar.interval(0.95)[0], CI_Fvar.interval(0.95)[1], color = 'g', alpha = 0.5)

        except:
            CI68_Fvar = st.norm.interval(alpha=0.68, loc= Fvar_mean, scale=np.std(Fvar))
            CI95_Fvar = st.norm.interval(alpha=0.95, loc= Fvar_mean, scale=np.std(Fvar))
            ax3.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI68_Fvar[0], CI68_Fvar[1], color = 'g', alpha = 0.5)
            ax3.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI95_Fvar[0], CI95_Fvar[1], color = 'g', alpha = 0.5)


        ax3.set_ylabel(r"$F_{var}$")
        ax3.minorticks_on()
        ax3.tick_params(axis='x', which = 'major', top = True, direction= 'in', length = 8)
        ax3.tick_params(axis='x', which = 'minor', top = True, direction= 'in', length = 4)
        ax3.set_xlim(xmin = binneddf[0].min()-1, xmax = binneddf[0].max()+1)
        ax3.legend(loc = 'upper right')
        ax3.set_xlabel('Time (BTJD - days)')

        if sectbysect == True:
            plt.savefig(subdir+'/'+target+'_excess_rms_varianceCI_'+bin_time+'hr_sectors'+group+'.pdf', format = 'pdf',bbox_inches='tight') 
        else:
            plt.savefig(subdir+'/'+target+'_excess_rms_varianceCI_'+bin_time+'hr.pdf', format = 'pdf',bbox_inches='tight')
        #                 plt.show()
        plt.close()

    except:
        plt.close()
        try:
            fig = plt.figure(figsize=(8,4))

            plt.errorbar(binneddf[0], Xvar, yerr = binneddf[1]*XvarErr, linestyle ="None", fmt='+', markersize = 4, color = 'k')
            plt.axhline(Xvar_mean, color = 'k', linestyle = 'solid', label = r'$\langle σ_{XS}^{2} \rangle$ = %.2e'%Xvar_mean)
            try:
                try:
                    unique, counts = np.unique(Xvar, return_counts=True)
                    pk = [x/len(Xvar) for x in counts]
                    CI_Xvar = st.rv_discrete(a = -np.inf,values=(unique, pk))  
                except:
                    CI_Xvar = st.rv_discrete(a = -np.inf,values=(Xvar, np.array(len(Xvar)*[1/len(Xvar)])))         
                plt.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI_Xvar.interval(0.68)[0], CI_Xvar.interval(0.68)[1], color = 'g', alpha = 0.5)
                plt.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI_Xvar.interval(0.95)[0], CI_Xvar.interval(0.95)[1], color = 'g', alpha = 0.5)

            except:
                CI68_Xvar = st.norm.interval(alpha=0.68, loc= Xvar_mean, scale=np.std(Xvar))                
                CI95_Xvar = st.norm.interval(alpha=0.95, loc= Xvar_mean, scale=np.std(Xvar))
                plt.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI68_Xvar[0], CI68_Xvar[1], color = 'g', alpha = 0.5)
                plt.fill_between([binneddf[0].min()-1,binneddf[0].max()+1], CI95_Xvar[0], CI95_Xvar[1], color = 'g', alpha = 0.5)

            plt.minorticks_on()
            plt.tick_params(axis='x', which = 'major', top = True, direction= 'in', length = 8)
            plt.tick_params(axis='x', which = 'minor', top = True, direction= 'in', length = 4)
            plt.legend(loc = 'upper right')

            plt.xlabel('Time (BTJD - days)')
            plt.ylabel(r"$σ_{XS}^{2}$")

            plt.xlim(xmin = binneddf[0].min()-1, xmax = binneddf[0].max()+1)

            if sectbysect == True:
                plt.title(target+r': $σ_{XS}^{2}$, '+bin_time+'hr bins sectors '+group,fontsize='x-large')
                plt.savefig(subdir+'/'+target+'_excess_rms_varianceCI_'+bin_time+'hr_sectors'+group+'.pdf', format = 'pdf',bbox_inches='tight') 
            else:
                plt.title(target+r': $σ_{XS}^{2}$, '+bin_time+'hr bins',fontsize='x-large')
                plt.savefig(subdir+'/'+target+'_excess_rms_varianceCI_'+bin_time+'hr.pdf', format = 'pdf',bbox_inches='tight')

        #                     plt.show()
            plt.close()

        except:
            plt.close()
            print("I was givin' 'er all she's got Cap'n")

    return()


###########################################################################################################
###########################################################################################################
'''
Definition of various functions regarding light curve modification 
including interpolation, stitching, and rebinning 
'''

def interpolate_data(target, df, pcm):
    '''
    Interpolation of light curve or light curve secgments
    
     inputs:
         target (string)        - Name of desired object to analyze
         df (DataFrame)         - dataframe of uninterpolated light curve
         pcm (int)              - specifier of regression pricipal component method
         
     outputs:
        interp_df (DataFrame)   - dataframe of interpolated light curve
        
    '''

    
    sampspace = np.round(df[0][1]-df[0][0],6)
    delt_crit = round_up_decimals(sampspace,5)
#     print(delt_crit)
    
                        
    time = [df[0][0]]
    flux = [df[1][0]]
    err = [df[2][0]]
  
    ## Sequence of finding gaps in data, linearly interpolating between gaps, and plotting of new figures.
                        
    for i in range(1,len(df[0])):
        
        delt = df[0][i] - df[0][i-1]
        
        if delt > delt_crit:
            buffer_bad = True
            pts = 15
                        
            early_time_buffer = []
            late_time_buffer = []
            
            early_flux_buffer = []
            late_flux_buffer = []

            while buffer_bad and pts >=1:
                try:
                    early_time_buffer = df[0][i-pts:i]
                    late_time_buffer = df[0][i:i+pts]

                    early_flux_buffer = df[1][i-pts:i]
                    late_flux_buffer = df[1][i:i+pts]
                    
                    buffer_bad = False
                
                    
                except:
                    pts -= 1
                    
            
            t_gap = np.arange(start = df[0][i-1] + sampspace, stop = df[0][i], step = sampspace)
            f_gap = np.interp(x = t_gap, xp = np.append(early_time_buffer, late_time_buffer), fp = np.append(early_flux_buffer,late_flux_buffer))
            e_gap = np.zeros(len(t_gap))
                               
            time.extend(np.array(t_gap))
            flux.extend(np.array(f_gap))
            err.extend(np.array(e_gap))
            
        else: 
            time.append(df[0][i])
            flux.append(df[1][i])
            err.append(df[2][i])
    

    interp_df = pd.DataFrame(np.column_stack((time,flux,err)),columns = None, index = None)
    
    interp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    interp_df = interp_df.dropna()
    
    if pcm == 1: 
        interp_df.to_csv(directory+'/'+target+'_cycle1_PCA_interpolated_lc.dat',sep = ' ',header = False, index = False)
    if pcm == 2: 
        interp_df.to_csv(directory+'/'+target+'_cycle1_simple_hybrid_interpolated_lc.dat',sep = ' ',header = False, index = False)
    if pcm == 3: 
        interp_df.to_csv(directory+'/'+target+'_cycle1_full_hybrid_interpolated_lc.dat',sep = ' ',header = False, index = False)

    
    return(interp_df)


###########################################################################################################

def bin_error(n, bins, flux):
    '''
    Calculate error on binned photometric flux
    
     inputs:
         n (array)        - bin counts
         bins (array)     - flux bins
         flux (array)     - flux values
         
     outputs:
        bin_err (array)   - error on flux bins
        
    '''
    
    bin_err = np.zeros(len(n))

    for i in range(0,len(n)):
        bin_flux = [x for x in flux if x >= bins[i] and x < bins[i+1]]
        bin_err[i] = np.std(bin_flux)/np.sqrt(n[i])
        
    return(bin_err)

###########################################################################################################

def rebinning(df,bin1,bin2):
    '''
    Rebin light curve under two user-specified schema (in hours)
    
     inputs:
        n (array)         - bin counts
        bin1 (string)     - string which specified how many hours for one binning scheme
        bin2 (string)     - string which specified how many hours for one binning scheme
        
     outputs:
        binnedflux_bin1 (DataFrame)   - data frame for binned flux under 
                                            specified binning scheme (time,flux,error)
        binnedflux_bin2 (DataFrame)   - data frame for binned flux under 
                                            specified binning scheme (time,flux,error)
        
    '''
    
    print("Rebinning.\n")
    # Fill gaps in time with NaN as flux values
    time, flux, err = df[0], df[1], df[2]
    
    ## Rebinning:
    
    time_bin1 = []
    time_bin2 = []
    
    binflux_bin1 = []
    binflux_bin2 = []
    
    binerr_bin1 = []
    binerr_bin2 = []
    
    binstd_bin1 = []
    binstd_bin2 = []

    binN_bin1 = []
    binN_bin2 = []
    
    binMSE_bin1 = []
    binMSE_bin2 = []
    

    for j in range(0,2):
        tmp_time = time[0]
        tmp_time_idx = 0
        
        tbin = [time[0]]
        fbin = [flux[0]]
        ebin = [err[0]]
        
        for i in range(1,len(time)):
                 
            if j == 0 and (time[i] - tmp_time) >= np.round((bin1*0.04166667),4): 
                
                temp_bin = np.column_stack((fbin,ebin))
                
                time_bin1.append(np.nanmean(tbin))
                binflux_bin1.append(np.nanmean([x for x,y in temp_bin if y != 0.0]))
                N = len([x for x,y in temp_bin if y != 0.0])
                binN_bin1.append(N)
                binerr_bin1.append(np.sqrt(np.nansum([y**2 for x,y in temp_bin if y != 0.0]))/N)
                binstd_bin1.append(np.nanstd([x for x,y in temp_bin if y != 0.0], ddof = 1))
                binMSE_bin1.append(np.nanmean([y**2 for x,y in temp_bin if y != 0.0]))
                
                
                tmp_time = time[i]
                tbin = [time[i]]
                fbin = [flux[i]]
                ebin = [err[i]]
                                
                 
            elif j == 1 and (time[i] - tmp_time) >= np.round((bin2*0.04166667),4): 
                
                temp_bin = np.column_stack((fbin,ebin))
                
                time_bin2.append(np.nanmean(tbin))
                binflux_bin2.append(np.nanmean([x for x,y in temp_bin if y != 0.0]))
                N = len([x for x,y in temp_bin if y != 0.0])
                binN_bin2.append(N)
                binerr_bin2.append(np.sqrt(np.nansum([y**2 for x,y in temp_bin if y != 0.0]))/N)
                binstd_bin2.append(np.nanstd([x for x,y in temp_bin if y != 0.0], ddof = 1))
                binMSE_bin2.append(np.nanmean([y**2 for x,y in temp_bin if y != 0.0]))
                
                
                tmp_time = time[i]
                tbin = [time[i]]
                fbin = [flux[i]]
                ebin = [err[i]]
                                
            elif i == len(time):
                
                if j == 0:
                    time_bin1.append(np.nanmean(tbin))
                    binflux_bin1.append(np.nanmean([x for x,y in temp_bin if y != 0.0]))
                    N = len([x for x,y in temp_bin if y != 0.0])
                    binN_bin1.append(N)
                    binerr_bin1.append(np.sqrt(np.nansum([y**2 for x,y in temp_bin if y != 0.0]))/N)
                    binstd_bin1.append(np.nanstd([x for x,y in temp_bin if y != 0.0], ddof = 1))
                    binMSE_bin1.append(np.nanmean([y**2 for x,y in temp_bin if y != 0.0]))
                    
                elif j == 1:
                    time_bin2.append(np.nanmean(tbin))
                    binflux_bin2.append(np.nanmean([x for x,y in temp_bin if y != 0.0]))
                    N = len([x for x,y in temp_bin if y != 0.0])
                    binN_bin2.append(N)
                    binerr_bin2.append(np.sqrt(np.nansum([y**2 for x,y in temp_bin if y != 0.0]))/N)
                    binstd_bin2.append(np.nanstd([x for x,y in temp_bin if y != 0.0], ddof = 1))
                    binMSE_bin2.append(np.nanmean([y**2 for x,y in temp_bin if y != 0.0]))
                    
            else:
                tbin.append(time[i])
                fbin.append(flux[i])
                ebin.append(err[i])
                                
                        

    binnedflux_bin1 = pd.DataFrame(np.column_stack((time_bin1, binflux_bin1, binerr_bin1, binstd_bin1,binN_bin1,binMSE_bin1)))
    binnedflux_bin2 = pd.DataFrame(np.column_stack((time_bin2, binflux_bin2, binerr_bin2, binstd_bin2,binN_bin2,binMSE_bin2)))

    binnedflux_bin1.replace([np.inf ,0.0, -np.inf], np.nan)
    binnedflux_bin2.replace([np.inf, 0.0, -np.inf], np.nan)
    binnedflux_bin1 = binnedflux_bin1.dropna()
    binnedflux_bin1 = binnedflux_bin1.reset_index(drop=True)
    binnedflux_bin2 = binnedflux_bin2.dropna()
    binnedflux_bin2 = binnedflux_bin2.reset_index(drop=True)
    


    return(binnedflux_bin1,binnedflux_bin2)


def lc_stitch(unstitched_lc):
    '''
    Stitch together multiple sectors of light curve if not already stitched
    
     inputs:
        unstitched_lc (array of arrays)   - multi-sector light curve
     outputs:
        full_lc_time (array)              - stitched light curve component
        full_lc_flux (array)              - stitched light curve component
        full_lc_err (array)               - stitched light curve component
    '''

    for j in range(0,len(unstitched_lc)):
        if j!=0:
            sector = str(j+1)

        lc = unstitched_lc[j]

        t = lc[:,0]
        f = lc[:,1]
        err = lc[:,2]


        if j == 0:

            full_lc_time = t
            full_lc_flux = f
            full_lc_err= err

        else:

            first_flux = np.mean(f[:10])
            last_flux = np.mean(full_lc_flux[-10:])

            scale_factor= first_flux - last_flux

            if scale_factor > 0:

                scaled_flux = f - abs(scale_factor)

            if scale_factor < 0:

                scaled_flux = f + abs(scale_factor)

            full_lc_time = np.append(full_lc_time,t)
            full_lc_flux = np.append(full_lc_flux,scaled_flux)
            full_lc_err = np.append(full_lc_err,err)

    return(full_lc_time,full_lc_flux,full_lc_err)

###########################################################################################################
###########################################################################################################

def lognormal_fit(n, flux, bins, bin_center, bin_err, i, directory, save_tail, bin1, bin2, visual = False):    
    '''
    Algorithm to fit distribution to lognormal curve
    
     inputs:
        n (array)                    - bin counts
        flux (array)                 - flux values
        bins (array)                 - flux bins
        bin_center (array)           - flux bin centers
        bin_err (array)              - bin errors
        i (int)                      - specifier for which version of light curve is being analyzed 
        directory (directory)        - The directory in which figures will be saved
        save_tail (string)           - tail of name assigned to file to be saved
        bin1 (string)                - string which specified how many hours for one binning scheme
        bin2 (string)                - string which specified how many hours for one binning scheme)
        visual (boolean)             - should figures be shown
    '''
    
    
    maxfreq = n.max()

    if len(str(int(bin_center[0] +0.5))) <= 2:
        tick_labels = [str(np.round(x,decimals=2)) for x in bin_center]
        
    elif len(str(int(bin_center[0] + 0.5))) == 3 :
        tick_labels = [str(np.round(x,decimals=1)) for x in bin_center]
        
    else:
        tick_labels = [str(int(x + 0.5)) for x in bin_center]

    
    plt.figure(figsize=(12,6))
    
    shape, loc, scale = st.lognorm.fit(np.array(flux), method = "MM")
    init_params = (maxfreq, np.log(scale), shape)
    
    lognorm_m = op.minimize(Min_PDF, [*init_params], args=(np.array((n,bins),dtype='object'),lognormal), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
    lognorm_pars = lognorm_m['x']
    
    plt.bar(bin_center, n, width = 0.8*(bin_center[1]-bin_center[0]), alpha=0.9, color= 'b', align='center',tick_label=tick_labels) # yerr=bin_err, 

       
    logbins = np.linspace(bins[0],bins[-1:],num=500)
    lognorm_fit = st.lognorm.pdf(logbins,shape,loc,scale)


    fit_max = np.max(lognorm_fit)
    fit_mean = loc

    plt.plot(logbins,lognorm_fit,color='r',label='log-normal fit')#\n$χ^{2}: %.2e, p: %.2f$'%(lognorm_chi2, lognorm_p))
    plt.axvline(fit_mean,color= 'k',linestyle = 'dashed')

    plt.grid(axis='y', alpha=0.75)
   
    plt.suptitle('Flux distibution: '+target) 
    plt.xlabel('Flux bins')
    plt.ylabel('Normalized N')
    
    if fit_mean >= bins[-1:][0] or fit_mean <= bins[0]: 
        # Set a clean x-axis limit.
        plt.xlim(xmin = bins[0], xmax = bins[-1:][0])
    
    # Set a clean upper y-axis limit.
    plt.ylim(ymax= 1.15*maxfreq if maxfreq>fit_max else 1.15*fit_max)
    plt.legend()


    if len(n) < 15:
        plt.xticks(rotation=30)
    else:
        plt.xticks(rotation=90)
    
    if i==2:
        plt.title(bin1+" hr bins")
        plt.savefig(directory+'/'+target+'_flux_dist_lognorm_'+bin1+'hr'+save_tail, format = 'pdf',bbox_inches='tight')


    if i==3:
        plt.title(bin2+" hr bins")
        plt.savefig(directory+'/'+target+'_flux_dist_lognorm_'+bin2+'hr'+save_tail, format = 'pdf',bbox_inches='tight')


    if visual == True:
        plt.show()
        
    plt.close()
    return()

###########################################################################################################

def bimodal_fit(n, flux, bins, bin_center, bin_err, i, directory, save_tail, bin1, bin2, visual = False, init_params = None):
    '''
    Algorithm to fit distribution to bimodal normal curve
    
     inputs:
        n (array)                    - bin counts
        flux (array)                 - flux values
        bins (array)                 - flux bins
        bin_center (array)           - flux bin centers
        bin_err (array)              - bin errors
        i (int)                      - specifier for which version of light curve is being analyzed 
        directory (directory)        - The directory in which figures will be saved
        save_tail (string)           - tail of name assigned to file to be saved
        bin1 (string)                - string which specified how many hours for one binning scheme
        bin2 (string)                - string which specified how many hours for one binning scheme)
        visual (boolean)             - should figures be shown
    '''
    
    flux = sorted(flux)
    
    if len(str(int(bin_center[0] +0.5))) <= 2:
        tick_labels = [str(np.round(x,decimals=2)) for x in bin_center]
        
    elif len(str(int(bin_center[0] + 0.5))) == 3 :
        tick_labels = [str(np.round(x,decimals=1)) for x in bin_center]
        
    else:
        tick_labels = [str(int(x + 0.5)) for x in bin_center]

    plt.figure(figsize=(12,6))
    
    plt.bar(bin_center,n, width = 0.8*(bin_center[1]-bin_center[0]), alpha=0.9, color= 'b', align='center',tick_label=tick_labels) # yerr=bin_err,
    


    if init_params == None:
        loc1, scale1 = st.norm.fit(flux[:roundedfractionalrange(flux, np.divide(2,3))])
        loc2, scale2 = st.norm.fit(flux[-roundedfractionalrange(flux, np.divide(2,3)):])
        init_params = (0.5, sorted(n)[-1:][0], loc1, scale1, sorted(n)[-4:][0], loc2, scale2)
    
    for j in range(0,len(init_params)):
        if np.isnan(init_params[j]) == True:
            print("nan at init_params[%i]"%j)

    bimod_m = op.minimize(Min_PDF, [*init_params], args=(np.array((n,bins),dtype='object'),bimodal), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
    bimod_pars = bimod_m['x']
    
    fit_flux = np.linspace(bins[0],bins[-1:],num=500) 
    bimod_fit = bimodal(fit_flux, *bimod_pars)
    
    fit_max = np.max(bimodal(fit_flux,*bimod_pars))
    maxfreq = n.max()
    
    mu1 = bimod_pars[2]
    mu2 = bimod_pars[5]
    
    plt.plot(fit_flux, bimodal(fit_flux,*bimod_pars), color='r', label='bimodal fit')#\n$χ^{2}: %.2e, p: %.2f$'%(bimod_chi2, bimod_p))
    plt.axvline(mu1,color= 'k',linestyle = 'dashed', label = '$μ_{1}$')
    plt.axvline(mu2,color= 'k',linestyle = 'dotted', label = '$μ_{2}$')
    
    
    plt.grid(axis='y', alpha=0.75)
    plt.suptitle('Flux distibution: '+target) 
    plt.xlabel('Flux bins')
    plt.ylabel('Normalized N')
    
    if mu1 >= bins[-1:][0] or mu1 <= bins[0] or mu2 >= bins[-1:][0] or mu2 <= bins[0]:
        # Set a clean x-axis limit.
        plt.xlim(xmin = bins[0], xmax = bins[-1:][0])
        
    # Set a clean upper y-axis limit.
    plt.ylim(ymax= 1.15*maxfreq if maxfreq>fit_max else 1.15*fit_max)
    plt.legend()
    ticks = [int(np.round(x)) for x in bin_center]

    ## Try to make sure all tick labels show and do not overlap        
    if len(n) < 15:
        plt.xticks(rotation=30)
    else:
        plt.xticks(rotation=90)


    if i==2:
        plt.title(bin1+" hr bins")
            
        plt.savefig(directory+'/'+target+'_flux_dist_bimodal_'+bin1+'hr'+save_tail, format = 'pdf',bbox_inches='tight')

    if i==3:
        plt.title(bin2+" hr bins")
            
        plt.savefig(directory+'/'+target+'_flux_dist_bimodal_'+bin2+'hr'+save_tail, format = 'pdf',bbox_inches='tight')

    
    if visual == True:
        plt.show()
    plt.close()

    return()


##################################################################################################


def hist_and_fit(flux, i, subdir, target, sectbysect, group, bin1=0., bin2=0., visual = False):
    '''
    Algorithm to fit distribution to gaussian, bimodal gaussian, and lognormal curves
    
     inputs:
        flux (array)                 - flux values
        i (int)                      - specifier for which version of light curve is being analyzed 
        subdir (directory)           - The directory in which figures will be saved
        target (string)              - target name
        sectbysect (boolean)         - sector by sector analysis or not (default is False)
        group (string)               - current sectors under analysis if sectbysect is True 
        bin1 (string)                - string which specified how many hours for one binning scheme
        bin2 (string)                - string which specified how many hours for one binning scheme)
        visual (boolean)             - should figures be shown
    '''

    if 0 < i < 4:
        nonzeroflux = np.nonzero(np.array(flux))[0]
        flux = np.array(uninterp_df[1][nonzeroflux])

    ## Estimate initial statistical parameters then remove outliers beyond 5-sigma
    try:
        flux_mean, flux_stdev = st.norm.fit(np.array(flux), method = "MM")
#         flux_mean = np.mean(flux)
#         flux_stdev = np.std(flux)
        flux_min = np.min(flux)
        flux_max = np.max(flux)

        fivesig = 5.0*flux_stdev
        flux = [x for x in flux if np.abs(flux_mean - x) <= fivesig]

        ## Reestimate statistical parameters without outliers
        flux_mean, flux_stdev = st.norm.fit(np.array(flux), method = "MM")
        flux_min = np.min(flux)
        flux_max = np.max(flux)
        
    except:
#         print(flux)
        ## Estimate statistical parameters without outliers
        flux_mean, flux_stdev = st.norm.fit(np.array(flux), method = "MM")
        flux_min = np.min(flux)
        flux_max = np.max(flux)
        
    
    ###############################################################
    
    ## Make histograms of data with and without normalization
    n, bins = np.histogram(flux, bins=OptBins(flux), range=(flux_min,flux_max), density=True)
    m, bims = np.histogram(flux, bins=OptBins(flux), range=(flux_min,flux_max), density=False)

    
    bin_err = np.divide(bin_error(m, bims, flux),np.sum(m))
#     bin_err = bin_error(m, bims, flux)
#     bin_rms = RMS(m, bims, flux)
      
    bin_center = bins[:-1] + np.diff(bins) / 2
    maxfreq = n.max()
    

    if len(str(int(bin_center[0] +0.5))) <= 2:
        tick_labels = [str(np.round(x,decimals=2)) for x in bin_center]
        
    elif len(str(int(bin_center[0] + 0.5))) == 3 :
        tick_labels = [str(np.round(x,decimals=1)) for x in bin_center]
        
    else:
        tick_labels = [str(int(x + 0.5)) for x in bin_center]
        
    ###############################################################
    
    gauss_fit_good = False
    
    while gauss_fit_good == False:
        
        ## Obtain statistical parameters for assumed Gaussian fit
        init_params = (n.max(), flux_mean, flux_stdev)
        try:
            gauss_pars, _ = curve_fit(gaussian, xdata=bin_center, ydata=n, p0=init_params, sigma=bin_err, maxfev= 1000)#, absolute_sigma = True)
        
            gauss_m = op.minimize(Min_PDF, [*gauss_pars], args=(np.array((n,bins),dtype='object'),gaussian), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
        except:
            gauss_m = op.minimize(Min_PDF, [*init_params], args=(np.array((n,bins),dtype='object'),gaussian), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
            
        gauss_pars = gauss_m['x']
        
        fit_mean = gauss_pars[1]
        fit_stdev = np.abs(gauss_pars[2])
                
        fit_FWHM = 2.0*np.sqrt(2.0*np.log(2.0))*fit_stdev ## assuming gaussian distribution

        plt.figure(figsize=(12,6), constrained_layout=True)
        # plt.figure(figsize=(15,6))

        ## Plot flux histogram as bar chart for special width and error bars
        plt.bar(bin_center, n, width = 0.85*(bin_center[1]-bin_center[0]), alpha=0.9, color= 'b', align='center',tick_label=tick_labels) #yerr=bin_err,


        ## Plot Gaussian fit over histogram
        fit_flux = np.linspace(bins[0],bins[-1:],num=500)
        
        gaussian_fit = gaussian(fit_flux, *gauss_pars)
        fit_max = np.max(gaussian_fit)
        plt.plot(fit_flux,gaussian_fit,color='r',label='Gaussian fit')#\n$χ^{2}: %.2e, p: %.2f$'%(gauss_chi2, gauss_p))
        
        
        ## Mark mean
        plt.axvline(fit_mean,color= 'k',linestyle = 'dashed', label = 'Mean = %.3e\nσ = %.3e'%(fit_mean,fit_stdev))

        ## Mark mean +/- stddev
        if fit_mean+fit_stdev <= flux_max:
            plt.axvline(fit_mean+fit_stdev,color= 'k',linestyle = 'dotted')
            plt.annotate(text='', xy=(fit_mean,1.05*maxfreq), xytext=(fit_mean+fit_stdev,1.05*maxfreq), arrowprops=dict(arrowstyle='<->'))
        else:
            plt.axvline(fit_mean-fit_stdev,color= 'k',linestyle = 'dotted')
            plt.annotate(text='', xy=(fit_mean,1.05*maxfreq), xytext=(fit_mean-fit_stdev,1.05*maxfreq), arrowprops=dict(arrowstyle='<->'))

        plt.legend()

        ###############################################################
        ## Create or save to new flux distributions directory
        try:
            directory = os.path.join(subdir, 'flux_distributions')
            os.mkdir(directory)

            print("Directory '%s' created\n" %directory)    
        except FileExistsError:
            directory = subdir+'/flux_distributions'
            
        if sectbysect == True:
            save_tail = '_sectors'+group+'.pdf'
            try:
                directory = os.path.join(subdir, 'flux_distributions/Sector_by_sector/')
                os.mkdir(directory)

                print("Directory '%s' created\n" %directory)    
            except FileExistsError:
                directory = subdir+'/flux_distributions/Sector_by_sector/'
        else:
            save_tail = '.pdf'

        ###############################################################

        if i < 4 :
            if fit_mean+fit_stdev <= flux_max:
                plt.text(fit_mean+0.45*fit_stdev,1.07*maxfreq,'σ$_{F}$',fontsize='medium')
            else:
                plt.text(fit_mean-0.45*fit_stdev,1.07*maxfreq,'σ$_{F}$',fontsize='medium')

            plt.suptitle('Flux distibution: '+target)
            plt.xlabel('Flux bins')

        elif i >= 4 :
            if fit_mean+fit_stdev <= flux_max:
                plt.text(fit_mean+0.45*fit_stdev,1.07*maxfreq,'σ$_{ΔF}$',fontsize='medium')
            else:
                plt.text(fit_mean-0.45*fit_stdev,1.07*maxfreq,'σ$_{ΔF}$',fontsize='medium')

            plt.suptitle('Subsequent Flux distibution: '+target)
            plt.xlabel('ΔF$_{ij}$')

        ## Try to make sure all tick labels show and do not overlap        
        if len(n) < 15:
            plt.xticks(rotation=30)
        else:
            plt.xticks(rotation=90)

        plt.grid(axis='y', alpha=0.75)
        plt.ylabel('Normalized N')

        if fit_mean >= bins[-1:][0] or fit_mean <= bins[0]:
            # Set a clean x-axis limit.
            plt.xlim(xmin = bins[0], xmax = bins[-1:][0])
        
        # Set a clean upper y-axis limit.
        plt.ylim(ymax = 1.15*maxfreq if maxfreq > fit_max else 1.15*fit_max)

        # Set a clean 5-sigma x-axis limit.
    #     fivesig = 5.0*fit_stdev
    #     plt.xlim(xmin = (fit_mean - 1.1*fivesig) if np.abs(fit_mean - np.min(bin_center)) > fivesig else (np.min(bin_center) - 0.1*np.abs(np.min(bin_center))),\
    #         xmax = (fit_mean + 1.1*fivesig) if np.abs(np.max(bin_center) - fit_mean) > fivesig else (np.max(bin_center) + 0.1*np.abs(np.max(bin_center))) )


        if i == 0:
            plt.title("Raw light curve, unbinned")
            plt.savefig(directory+'/'+target+'_raw_flux_dist_gauss_unbinned'+save_tail, format = 'pdf')

        if i == 1:
            plt.title("Quaver regression, unbinned")
            plt.savefig(directory+'/'+target+'_flux_dist_gauss_unbinned'+save_tail, format = 'pdf')

        if i == 2:
            plt.title("Quaver regression, "+bin1+" hr bins")
            plt.savefig(directory+'/'+target+'_flux_dist_gauss_'+bin1+'hr'+save_tail, format = 'pdf')

        if i == 3:
            plt.title("Quaver regression, "+bin2+" hr bins")
            plt.savefig(directory+'/'+target+'_flux_dist_gauss_'+bin2+'hr'+save_tail, format = 'pdf')



        if i == 4:
            plt.title("Raw light curve, unbinned")
            plt.savefig(directory+'/'+target+'_raw_subflux_dist_unbinned'+save_tail, format = 'pdf')

        if i == 5:
            plt.title("Quaver regression, unbinned")
            plt.savefig(directory+'/'+target+'_subflux_dist_unbinned'+save_tail, format = 'pdf')

        if i == 6:
            plt.title("Quaver regression, "+bin1+" hr bins")
            plt.savefig(directory+'/'+target+'_subflux_dist_'+bin1+'hr'+save_tail, format = 'pdf')

        if i == 7:
            plt.title("Quaver regression, "+bin2+" hr bins")
            plt.savefig(directory+'/'+target+'_subflux_dist_'+bin2+'hr'+save_tail, format = 'pdf')


#         visual = True
        if visual == True:
            plt.show()
        
            fit_check = input("Is gaussian fit satisfactory? [y/n] ")

            if fit_check == "y" or fit_check == "Y" or fit_check == "yes" or fit_check == "YES":
                gauss_fit_good = True
            else: 
                continue
        else:
            gauss_fit_good = True
        
        plt.close()

    if i > 1 and i < 4 :
        lognorm_fit_good = False
        while lognorm_fit_good == False:
            
            
            lognormal_fit(n, flux, bins, bin_center, bin_err, i, directory, save_tail, bin1, bin2, visual = False)
            
            if visual == True:
            
                fit_check = input("Is lognormal fit satisfactory? [y/n] ")
            
                if fit_check == "y" or fit_check == "Y" or fit_check == "yes" or fit_check == "YES":
                    lognorm_fit_good = True
                else: 
                    continue
                plt.close()
                
        
            else:
                lognorm_fit_good = True
                plt.close()
        
        
        bimodal_fit_good = False
        while bimodal_fit_good == False:
            
            bimodal_fit(n, flux, bins, bin_center, bin_err, i, directory, save_tail, bin1, bin2, visual = False)
            
            if visual == True:
                fit_check = input("Is bimodal fit satisfactory? [y/n] ")
            
                if fit_check == "y" or fit_check == "Y" or fit_check == "yes" or fit_check == "YES":
                    bimodal_fit_good = True
                else: 
                    continue
                plt.close()
                
            else:
                bimodal_fit_good = True
                plt.close()
                
            

    return(fit_stdev, flux_mean, flux_min, flux_max, bin_center)



###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

## Begin by setting the driectory in which the target files are kept as rootdir

## Choose sample "analysis" or "removed"
sample = "analysis"

if sample == "analysis":
    rootdir = '/users/rdingler/AGNstudy/LightCurves/Analysis/'
if sample == "removed":
    rootdir = '/users/rdingler/AGNstudy/LightCurves/Removed_sample/'

## User should specficy whether thay would like a secondary data sheet with user-specified modification
## Specify modifications at bottom of program
## May prove useful for very large or very small numbers
adj = True  


## Set cycle number and Pricipal Component Method
# pcm == 1: principal component analysis
# pcm == 2: simple hybrid method
# pcm == 3: full hybrid method

cycle = '1'
pcm = 2

if pcm == 1: 
    file_tail = '_cycle'+cycle+'_PCA_lc.dat'
if pcm == 2: 
    file_tail = '_cycle'+cycle+'_hybrid_lc.dat'
if pcm == 3: 
    file_tail = '_cycle'+cycle+'_full_hybrid_lc.dat'

## Set binning schema for binned light curve analysis in hours
bin1 = '6' #hrs
bin2 = '12' #hrs 

## Set some specific designation which will be used later to identify the spliced light curves as output by quaver
## This will pick these files for use if provided. If not available, say, in the case of sector by sector analysis
## wherein this file is the etire light curve, this will help the algorithm to know that it needs to collect and 
## synthesize each sector as specified by the user. User should write use the definition above for "lc_stitch"
## to manually make stitched light curves for their desired sector groupings if not individual.
    
file_mid = '_spliced_lightcurve_sectors_'

## Counting as preventative measure when only wanting to do a limited number of files. Disable if all files should be read.
count = 0

for subdir, dirs , files in os.walk(rootdir):
    
    count += 1
    # if count == 2:
    #     sys.exit()
     
    ## Get target name from subdirectory name
    target = os.path.basename(subdir)
    
        
    ## Initialize arrays for plotting
    for file in files:
        sectbysect = False
#          rmv_file_denote = ''
#         if file.endswith('rmv_file_denote'):  ## lines only included in case of cleanup
#             os.remove(os.path.join(subdir,file))

        if file.endswith(file_tail) or file_mid in file:
            ## When walking through directories, make sure to specify which directory names are not to be included
            ## User should make sure this method works for their arrangement of quaver output files and any other
            ## preliminary data stored in the same directory as the quaver output
            if target != "" and target != "eleanor_apertures" and target != "quaver_lc_apertures" and \
                target != "flux_distributions" and target != "Preliminary_tests" and target != "LCs_forpub" and\
                    target != ".ipynb_checkpoints" and target != "Sector_by_sector":
                print("Target: %s"%target)
                
            print("\nWorking in directory: %s\n"%subdir)
            
            
            
            group = ''
            
            try:
                if file_mid in file: ## if True, this object is designated for sector by sector analysis by user-specified designation
                    sectbysect = True
                    unstitched_lc = []

                    group = file.partition('sectors_')[2].partition('.dat')[0]
                    print("Analyzing spliced sectors "+group+" from file '%s"%file+"'")
                    sectors = np.array([s for s in group.split('+')])
                    for sector in sectors:
                        raw_df = pd.read_table(os.path.join(subdir, target+'_cycle'+cycle+'_sector'+sector+'_raw_lc.dat'), sep = ' ', header=None, dtype='float')
                        unstitched_lc.append(np.column_stack((raw_df[0],raw_df[1],raw_df[2])))

                    t_raw, f_raw, e_raw = lc_stitch(unstitched_lc)
                    raw_df = pd.DataFrame(np.column_stack((t_raw,f_raw,e_raw)),columns = None, index = None)

                else:
                    print("Analyzing full light curve from file '%s"%file+"'")
                    raw_df = pd.read_table(os.path.join(subdir, target+'_full_raw_lc.dat'), sep = ' ', header=None, dtype='float')
                raw = True
            except:
                raw = False
                    
            check_col_label(rootdir,str(bin1),str(bin2),sample = sample, raw = raw)

            if sample == "analysis":
                lc_data = open(os.path.join(rootdir,'LC_variability_analysis.csv') , "a")
            if sample == "removed":
                lc_data = open(os.path.join(rootdir,'LC_variability_rmv.csv') , "a")
                
            writer = csv.writer(lc_data)

            uninterp_df = pd.read_table(os.path.join(subdir, file), sep = ' ', header=None, dtype='float')
            
            
###############################################################################################             
## Find shortest timescales of exponential rise and decay
            print("\nFinding shortest time scales of variablity. This may take a while for light light curves.")        
            tau_dec_min, delt_dec_min, tau_inc_min, delt_inc_min = shortest_timescale(rootdir,target,uninterp_df,sectbysect,sectors=group)
            
            
############################################################################################### 
## Find (Smith 2018) statistics and check flux-change distribution, 
## as well as comparing the flux-change in the unbinned raw vs the unbinned corrected data

            df = interpolate_data(target,uninterp_df)
            binnedflux_bin1, binnedflux_bin2 = rebinning(df,float(bin1),float(bin2))
            
            print("\nFitting histograms of flux data")
            for i in range(0,4):
                if i == 0 and raw == True:
                    lc_raw_stdev, flux_raw_mean, flux_raw_min , flux_raw_max, flux_raw_bins= hist_and_fit(raw_df[1],i,subdir,target,sectbysect,group)
                if i == 1:
                    lc_reg_stdev, flux_reg_mean, flux_reg_min , flux_reg_max, flux_reg_bins = hist_and_fit(uninterp_df[1],i,subdir,target,sectbysect,group)
                if i == 2:
                    lc_bin1_stdev, flux_bin1_mean, flux_bin1_min , flux_bin1_max, flux_bin1_bins = hist_and_fit(binnedflux_bin1[1],i,subdir,target,sectbysect,group,bin1=bin1,bin2=bin2)
                if i == 3:
                    lc_bin2_stdev, flux_bin2_mean, flux_bin2_min , flux_bin2_max, flux_bin2_bins = hist_and_fit(binnedflux_bin2[1],i,subdir,target,sectbysect,group,bin1=bin1,bin2=bin2)

#######################################################################################                    
## Find F-parameters
## Potentially useful staistic but not in this case

#             F_param_reg = lc_reg_stdev**2 / np.nanmean(df[2]**2)
#             F_param_bin1 = lc_bin1_stdev**2 / np.nanmean(binnedflux_bin1[2])
#             F_param_bin2 = lc_bin2_stdev**2 / np.nanmean(binnedflux_bin2[2])


#######################################################################################
## Excess variance and fractional rms amplitude calculations for intrabin data
            
            print("\nCalculating excess variance within %s hr and %s hr bins"%(bin1,bin2))
            Xvar_bin1, Fvar_bin1, XvarErr_bin1, FvarErr_bin1 = excess_variance(binnedflux_bin1[0],binnedflux_bin1[1],binnedflux_bin1[2],\
                                                                            stdev = binnedflux_bin1[3], len_lc = binnedflux_bin1[4], MSE = binnedflux_bin1[5] , total = False)
            
            Xvar_bin1_mean = np.nanmean(Xvar_bin1)
            Fvar_bin1_mean = np.nanmean(Fvar_bin1)
            
            plot_excess_variance(subdir, df, binnedflux_bin1, Xvar_bin1, XvarErr_bin1, Xvar_bin1_mean, Fvar_bin1, FvarErr_bin1, Fvar_bin1_mean, bin1, sectbysect, group)
            
            
            
            ##########################################################################
            ##########################################################################


            Xvar_bin2, Fvar_bin2, XvarErr_bin2, FvarErr_bin2 = excess_variance(binnedflux_bin2[0],binnedflux_bin2[1],binnedflux_bin2[2],\
                                                                            stdev = binnedflux_bin2[3], len_lc = binnedflux_bin2[4], MSE = binnedflux_bin2[5] , total = False)

            Xvar_bin2_mean = np.nanmean(Xvar_bin2)
            Fvar_bin2_mean = np.nanmean(Fvar_bin2)
            
            
            plot_excess_variance(subdir, df, binnedflux_bin2, Xvar_bin2, XvarErr_bin2, Xvar_bin2_mean, Fvar_bin2, FvarErr_bin2, Fvar_bin2_mean, bin2, sectbysect, group)
                  
                    
            ##############################################################################################################################
            ## Plot and save resultant RMS-Flux relation 
            print("\nCalculating RMS-Flux relation within %s hr and %s hr bins"%(bin1,bin2))
        
            idx = np.isfinite(binnedflux_bin1[1]) & np.isfinite(Fvar_bin1)
            rms_flux = np.array((binnedflux_bin1[1][idx], Fvar_bin1[idx], FvarErr_bin1[idx]))
            
            try:
                line1 = np.polyfit(rms_flux[0], rms_flux[1], 1)
                plt.errorbar(rms_flux[0], rms_flux[1], yerr= rms_flux[2], color='r', label=bin1+'hr bins, α = %.2e'%line1[0], alpha=0.8, linestyle ="None", fmt='.', markersize = 8)
                
                x1 = np.linspace(rms_flux[0].min(), rms_flux[0].max(), num=200)
                plt.plot(x1,linear(x1,*line1), color='r', linestyle='dotted')
            except: 
                try:
                    line1 = op.minimize(Min_PDF, [1.,0.], args=(np.array((rms_flux[1],rms_flux[0]),dtype='object'),linear), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
                    line1 = line1['x']
                    
                    plt.errorbar(rms_flux[0], rms_flux[1], yerr= rms_flux[2], color='r', label=bin1+'hr bins, α = %.2e'%line1[0], alpha=0.8, linestyle ="None", fmt='.', markersize = 8)
                    x1 = np.linspace(rms_flux[0].min(), rms_flux[0].max(), num=200)
                    plt.plot(x1,linear(x1,*line1), color='r', linestyle='dotted')
                except:    
                    line1 = [0.0,0.0]
                
            

            idx = np.isfinite(binnedflux_bin2[1]) & np.isfinite(Fvar_bin2)
            rms_flux = np.array((binnedflux_bin2[1][idx], Fvar_bin2[idx], FvarErr_bin2[idx]))

            try:
                line2 = np.polyfit(rms_flux[0], rms_flux[1], 1)
                plt.errorbar(rms_flux[0], rms_flux[1], yerr= rms_flux[2], color='b', label=bin2+'hr bins, α = %.2e'%line2[0], alpha=0.8, linestyle ="None", fmt='.', markersize = 8)
                x2 = np.linspace(rms_flux[0].min(), rms_flux[0].max(), num=200)
                plt.plot(x2,linear(x2,*line2), color='b', linestyle='dotted')
            except: 
                try:
                    line2 = op.minimize(Min_PDF, [1.,0.], args=(np.array((rms_flux[1],rms_flux[0]),dtype='object'),linear), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
                    line2 = line2['x']
                    plt.errorbar(rms_flux[0], rms_flux[1], yerr= rms_flux[2], color='b', label=bin2+'hr bins, α = %.2e'%line2[0], alpha=0.8, linestyle ="None", fmt='.', markersize = 8)
                    x2 = np.linspace(rms_flux[0].min(), rms_flux[0].max(), num=200)
                    plt.plot(x2,linear(x2,*line2), color='b', linestyle='dotted')
                except:    
                    line2 = [0.0,0.0]

            

            try:
                plt.xlabel("Average Flux of Bin (cts s$^{-1}$)")
                plt.ylabel("RMS (cts s$^{-1}$)")
                plt.legend()

                plt.title("RMS-flux relation")
                plt.savefig(subdir+'/'+target+'_rms_flux_dist.pdf', format = 'pdf',bbox_inches='tight')

    #             plt.show()
                plt.close()
            except:
                plt.close()
            
            ############################################################################################
            ############################################################################################
            
            ## Now a flux-binned version of the same relation
            try:
                idx = np.isfinite(binnedflux_bin1[1]) & np.isfinite(Fvar_bin1)
                rms_flux = np.array((binnedflux_bin1[1][idx], Fvar_bin1[idx], FvarErr_bin1[idx]))

                rms_n, rms_bins = np.histogram(rms_flux[0], bins = OptBins(rms_flux[0]), range=(rms_flux[0].min(),rms_flux[0].max()))
                bin_center = rms_bins[:-1] + np.diff(rms_bins) / 2

                rms_flux_binned = np.array((bin_center, np.zeros(len(bin_center)), np.zeros(len(bin_center))))

                for i in range(0,len(rms_bins)-1):
                    binleft = rms_bins[i]
                    binright = rms_bins[i+1]

                    temp_Fvar = []
                    temp_err = []

                    for j in range(0,len(rms_flux[0])):

                        if rms_flux[0][j] >= binleft and rms_flux[0][j] < binright:
                            temp_Fvar.append(rms_flux[1][j])
                            temp_err.append(rms_flux[2][j])
                        elif i == len(rms_bins)-1 and rms_flux[0][j] == rms_flux[0].max():
                            temp_Fvar.append(rms_flux[1][j])
                            temp_err.append(rms_flux[2][j])

                    rms_flux_binned[1][i] = np.nanmean(temp_Fvar)
                    rms_flux_binned[2][i] = np.sqrt(np.nansum(np.array(temp_err)**2)/len(temp_err)**2)

                try:
                    line1 = np.polyfit(rms_flux[0], rms_flux[1], 1)
                    plt.errorbar(rms_flux_binned[0], rms_flux_binned[1], yerr= rms_flux_binned[2], color='r', label=bin1+'hr bins, α = %.2e'%line1[0], alpha=0.8, linestyle ="None",\
                                 fmt='.', markersize = 8)
                    x1 = np.linspace(rms_flux_binned[0].min(), rms_flux_binned[0].max(), num=200)
                    plt.plot(x1,linear(x1,*line1), color='r', linestyle='dotted')
                except: 
                    try:
                        line1 = op.minimize(Min_PDF, [1.,0.], args=(np.array((rms_flux_binned[1],rms_bins),dtype='object'),linear), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
                        line1 = line1['x']
                        plt.errorbar(rms_flux_binned[0], rms_flux_binned[1], yerr= rms_flux_binned[2], color='r', label=bin1+'hr bins, α = %.2e'%line1[0], alpha=0.8, linestyle ="None",\
                                     fmt='.', markersize = 8)
                        x1 = np.linspace(rms_flux_binned[0].min(), rms_flux_binned[0].max(), num=200)
                        plt.plot(x1,linear(x1,*line1), color='r', linestyle='dotted')
                    except:    
                        line1 = [0.0,0.0]




                idx = np.isfinite(binnedflux_bin2[1]) & np.isfinite(Fvar_bin2)
                rms_flux = np.array((binnedflux_bin2[1][idx], Fvar_bin2[idx], FvarErr_bin2[idx]))

                rms_n, rms_bins = np.histogram(rms_flux[0], bins = OptBins(rms_flux[0]), range=(rms_flux[0].min(),rms_flux[0].max()))
                bin_center = rms_bins[:-1] + np.diff(rms_bins) / 2
                
                rms_flux_binned = np.array((bin_center, np.zeros(len(bin_center)), np.zeros(len(bin_center))))

                for i in range(0,len(rms_bins)-1):
                    binleft = rms_bins[i]
                    binright = rms_bins[i+1]

                    temp_Fvar = []
                    temp_err = []

                    for j in range(0,len(rms_flux[0])):

                        if rms_flux[0][j] >= binleft and rms_flux[0][j] < binright:
                            temp_Fvar.append(rms_flux[1][j])
                            temp_err.append(rms_flux[2][j])
                        elif i == len(rms_bins)-1 and rms_flux[0][j] == rms_flux[0].max():
                            temp_Fvar.append(rms_flux[1][j])
                            temp_err.append(rms_flux[2][j])

                    rms_flux_binned[1][i] = np.nanmean(temp_Fvar)
                    rms_flux_binned[2][i] = np.sqrt(np.nansum(np.array(temp_err)**2)/len(temp_err)**2)

                try:
                    line2 = np.polyfit(rms_flux[0], rms_flux[1], 1)
                    plt.errorbar(rms_flux_binned[0], rms_flux_binned[1], yerr= rms_flux_binned[2], color='b', label=bin2+'hr bins, α = %.2e'%line2[0], alpha=0.8, linestyle ="None", fmt='.', markersize = 8)
                    x2 = np.linspace(rms_flux_binned[0].min(), rms_flux_binned[0].max(), num=200)
                    plt.plot(x2,linear(x2,*line2), color='b', linestyle='dotted')
                except: 
                    try:
                        line2 = op.minimize(Min_PDF, [1.,0.], args=(np.array((rms_flux_binned[1],rms_bins),dtype='object'),linear), method='L-BFGS-B', options={'gtol':1e-6,'disp':False})
                        line2 = line2['x']
                        plt.errorbar(rms_flux_binned[0], rms_flux_binned[1], yerr= rms_flux_binned[2], color='b', label=bin2+'hr bins, α = %.2e'%line2[0], alpha=0.8, linestyle ="None", fmt='.', markersize = 8)
                        x2 = np.linspace(rms_flux_binned[0].min(), rms_flux_binned[0].max(), num=200)
                        plt.plot(x2,linear(x2,*line2), color='b', linestyle='dotted')
                    except:    
                        line2 = [0.0,0.0]

                try:
                    plt.xlabel("Average Flux of Bin (cts s$^{-1}$)")
                    plt.ylabel("RMS (cts s$^{-1}$)")
                    plt.legend()

                    plt.title("RMS-flux relation")
                    plt.savefig(subdir+'/'+target+'_rms_flux_binned_dist.pdf', format = 'pdf',bbox_inches='tight')

        #             plt.show()
                    plt.close()
                except:
                    plt.close()
                    
                    
            except:
                print("\nCould not make figures for binned rms-flux relation.\n")
                
                
            
#######################################################################################
 ## Excess variance and fractional rms amplitude calculations for itotal light curve
            print("\nCalculating excess variance of full light curves")
            

            NXvar_reg, Fvar_reg, NXvarErr_reg, FvarErr_reg = excess_variance(uninterp_df[0],uninterp_df[1],uninterp_df[2])
 
            NXvar_bin1, Fvar_bin1, NXvarErr_bin1, FvarErr_bin1 = excess_variance(binnedflux_bin1[0],binnedflux_bin1[1],binnedflux_bin1[2])

            NXvar_bin2, Fvar_bin2, NXvarErr_bin2, FvarErr_bin2 = excess_variance(binnedflux_bin2[0],binnedflux_bin2[1],binnedflux_bin2[2])
        
        
#######################################################################################
##Calculate subsequent flux distributions
                        
            delflux_raw = [y for x,y in np.column_stack((np.diff(raw_df[0]),np.diff(raw_df[1])))\
                           if x > 0.02 and x < 0.0209 ]


            delflux_reg = [y for x,y in np.column_stack((np.diff(uninterp_df[0]),np.diff(uninterp_df[1])))\
                           if x > 0.02 and x < 0.0209 ]

            
            delflux_bin1 = [y for x,y in np.column_stack((np.diff(binnedflux_bin1[0]),np.diff(binnedflux_bin1[1])))\
                           if x > 0.1 and x < 0.3 ]

            delflux_bin2 = [y for x,y in np.column_stack((np.diff(binnedflux_bin2[0]),np.diff(binnedflux_bin2[1])))\
                           if x > 0.3 and x < 0.7 ]
            

            print("\nFitting histograms of change in flux data")
            for i in range(4,8):
                if i == 4 and raw == True:
                    fit_raw_stdev, delflux_raw_mean, delflux_raw_min, delflux_raw_max, delflux_raw_bins = hist_and_fit(delflux_raw,i,subdir,target,sectbysect,group)
                if i == 5:
                    fit_reg_stdev, delflux_reg_mean, delflux_reg_min, delflux_reg_max, delflux_reg_bins = hist_and_fit(delflux_reg,i,subdir,target,sectbysect,group)
                if i == 6:
                    fit_bin1_stdev, delflux_bin1_mean, delflux_bin1_min, delflux_bin1_max, delflux_bin1_bins = hist_and_fit(delflux_bin1,i,subdir,target,sectbysect,group,bin1=bin1,bin2=bin2)
                if i == 7:
                    fit_bin2_stdev, delflux_bin2_mean, delflux_bin2_min, delflux_bin2_max, delflux_bin2_bins = hist_and_fit(delflux_bin2,i,subdir,target,sectbysect,group,bin1=bin1,bin2=bin2)
                
                
#######################################################################################
## Find the chi-squared per degree of freedom (Sesar 2007)
            print("\nCalculating chi-squared/dof")
            mean_dif = np.subtract(uninterp_df[1],np.nanmean(uninterp_df[1]))
            chi2_dof = (len(uninterp_df[1])-1)**-1 * np.sum(mean_dif**2/uninterp_df[2]**2)

            chi2, dof, chi2_dof = chisquare_per_dof(uninterp_df[1],np.mean(uninterp_df[1]),uninterp_df[2])

            
            
            
#######################################################################################
## Write and save results
            print("\nSaving data")
            if sectbysect == True:
                target_name = target+' sectors '+group
            else:
                target_name = target
           
            if raw == True:
                row = [target_name, flux_raw_mean, lc_raw_stdev, delflux_raw_mean, fit_raw_stdev,\
                    flux_reg_mean, flux_bin1_mean, flux_bin2_mean,\
                       lc_reg_stdev, lc_bin1_stdev, lc_bin2_stdev,\
                           delflux_reg_mean, delflux_bin1_mean, delflux_bin2_mean,\
                               fit_reg_stdev, fit_bin1_stdev, fit_bin2_stdev,\
                                    NXvar_reg, NXvarErr_reg, NXvar_bin1, NXvarErr_bin1, NXvar_bin2, NXvarErr_reg,\
                                        Fvar_reg, FvarErr_reg, Fvar_bin1, FvarErr_bin1, Fvar_bin2, FvarErr_bin2,\
                                            tau_inc_min, delt_inc_min, tau_dec_min, delt_dec_min,\
                                                chi2, dof, chi2_dof]
            
            if raw == False:
                row = [target, flux_reg_mean, flux_bin1_mean, flux_bin2_mean,\
                   lc_reg_stdev, lc_bin1_stdev, lc_bin2_stdev,\
                       delflux_reg_mean, delflux_bin1_mean, delflux_bin2_mean,\
                           fit_reg_stdev,fit_bin1_stdev, fit_bin2_stdev,\
                                Xvar_reg, Xvar_bin1_mean, Xvar_bin2_mean,\
                                    Fvar_reg, Fvar_bin1_mean, Fvar_bin2_mean,
                                        tau_inc_min, delt_inc_min, tau_dec_min, delt_dec_min,\
                                           chi2, dof, chi2_dof]

            writer.writerow(row)

            lc_data.close()
            print("\nMoving to next object")

            
            
if sample == "analysis":
    results = '/users/rdingler/AGNstudy/LightCurves/Analysis/LC_variability_analysis.csv'
if sample == "removed":
    results = '/users/rdingler/AGNstudy/LightCurves/Removed_sample/LC_variability_rmv.csv'

read_df = pd.read_csv(results)

if sample == "analysis":
    read_df.to_excel('/users/rdingler/AGNstudy/LightCurves/Analysis/LC_variability_analysis.xlsx',index=False) 
if sample == "removed":
    read_df.to_excel('/users/rdingler/AGNstudy/LightCurves/Removed_sample/LC_variability_rmv.xlsx',index=False)
    
if adj == True:            
    
    ## User-specified modicfications to data for potential ease of viewing
    print("\nAdjusting final data form")            
    

    for x in [y for y in read_df.columns.values if y != 'Target']:

        idx = np.isfinite(np.array(read_df[x], dtype = 'float64'))
        values = np.abs(np.array(read_df[x][idx], dtype = 'float64'))

        if x == 'σ^2_NXS(unbinned)' or x == 'err(σ^2_NXS)(unbinned)' or x == 'σ^2_NXS('+str(bin1)+'hr)' or x == 'err(σ^2_NXS)('+str(bin1)+'hr)' or x == 'σ^2_NXS('+str(bin2)+'hr)' or x == 'err(σ^2_NXS)('+str(bin2)+'hr)':

            read_df[x] = np.round(np.log10(read_df[x]),decimals = 4)
            new_column = '$log('+x+')'
            read_df = read_df.rename(columns={x:new_column})
        else:
            mean_median_middle = np.nanmean([np.nanmedian(np.log10([x for x in values if x!=0])),np.nanmean(np.log10([x for x in values if x!=0]))])
            power = int(mean_median_middle + 0.5)

            if power <= 1 and power >= -1:
                read_df[x] = np.round(read_df[x],decimals = 4)
            else:
                read_df[x] = np.round(read_df[x]/(10**power),decimals = 4)
                new_column = x +' [*10^'+str(power)+']'
                read_df = read_df.rename(columns={x:new_column})


    if sample == "analysis":
        read_df.to_csv('/users/rdingler/AGNstudy/LightCurves/Analysis/LC_variability_analysis_adjusted.csv',index=False)
        read_df.to_excel('/users/rdingler/AGNstudy/LightCurves/Analysis/LC_variability_analysis_adjusted.xlsx',index=False)

    if sample == "removed":
        read_df.to_csv('/users/rdingler/AGNstudy/LightCurves/Removed_sample/LC_variability_rmv_adjusted.csv',index=False)
        read_df.to_excel('/users/rdingler/AGNstudy/LightCurves/Removed_sample/LC_variability_rmv_adjusted.xlsx',index=False)

print("Done!")
