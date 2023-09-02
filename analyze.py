import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as tt
import graphviz

def main():
    
    #load json file to arviz inference data
    idata = az.from_json("results/inference data/idata_HBM.json")

    '''
    #populational level distribution
    eta = idata.posterior["mu"]
    for i in range(eta.shape[-1]):
        fig=az.plot_trace(eta[..., i])
        plt.savefig(f"results/parameters/population/eta {i+1}.png")
        plt.clf()
    
    #individual level distribution
    tau = idata.posterior["pho"]
    for i in range(tau.shape[-2]):
        for j in range(tau.shape[-1]):
            az.plot_trace(tau[...,i, j])
            plt.savefig(f"results/parameters/individual/subject {i+1} tau {j+1}.png")
            plt.clf()

    #combined individual level distribution
    for i in range(tau.shape[-1]):
            az.plot_trace(tau[...,i])
            plt.savefig(f"results/parameters/individual/combined/tau {i+1}.png")
            plt.clf()
    '''
    #theta correlation heat map 
    nPara=10
    iChain=0 #select the chain (three in total)
    iSubject=1 #subject index
    for i in range(nPara):
         # this loop travel through all theta combinations
         for j in range(i):
            #load theta trace
            theta_collection=idata.posterior['theta'].values[iChain]

            #specify theta[]
            a=theta_collection[:,iSubject,i]
            b=theta_collection[:,iSubject,j]

            #bound and step for heat map
            a_max, a_min, a_len = max(a), min(a), (max(a)- min(a))/100
            b_max, b_min, b_len = max(b), min(b), (max(b)- min(b))/100

            #plot heatmap
            corrcoef=np.corrcoef(a,b)[0,1].round(4)
            heatmap = plt.hist2d(a, b, bins = (np.arange(a_min,a_max,a_len),np.arange(b_min,b_max,b_len)), cmap = 'hot')
            plt.colorbar()
            plt.title(f'corrcoef={corrcoef}')
            plt.xlabel(f'theta{i+1}')
            plt.ylabel(f'theta{j+1}')
            plt.savefig(f"results/parameters/test/theta {i+1} & theta {j+1}.png")
            #clear label for current figure
            plt.clf()

    print('all done!')

if __name__ == '__main__':
    main()