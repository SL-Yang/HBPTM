import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as tt
import graphviz

def main():
    #number of trials, subjects, test
    nTrials=40
    nSubjects=18
    nTests=1
    nNoise=8
    nContrast=9
    nDraws=150000

    #read data
    my_data=pd.read_csv('data/data40q_var.csv')
    #contrast level , noise level, behaviral correct response
    c=tt.constant(my_data['Contrast']).reshape((nSubjects,nContrast*nNoise))
    n=tt.constant(my_data['Noise']).reshape((nSubjects,nContrast*nNoise))
    nCorrect=my_data[['nCorrect1','nCorrect2','nCorrect3','nCorrect4']]

    #uniform distribution lower and upper bound for 10 parameters
    lower=tt.constant([0,0,0,0,0,0,0,0,0,0])
    upper=tt.constant([1,1,10,3,2,1,1,10,2,2])

    #gamma distribution parameter alpha and beta with shape 1*10 population level
    sig_a=tt.ones(10)*5000
    sig_b=tt.ones(10)*3

    #gamma distribution parameter alpha and beta with shape 1*10 individual level
    del_a=tt.ones(10)*5000
    del_b=tt.ones(10)*3

    HBM=pm.Model()
    with HBM:
              
        mu=pm.Uniform('mu', lower=lower, upper=upper)
        
        sigma=pm.Gamma('sigma', alpha=sig_a, beta=sig_b)
        pho=pm.TruncatedNormal('pho', mu=tt.tile(mu,(nSubjects,1)), tau=sigma, lower=lower,upper=upper)
        
        delta=pm.Gamma('delta', alpha=del_a, beta=del_b)
        theta=pm.TruncatedNormal('theta', mu=tt.tile(pho,(nTests,1)), tau=delta, lower=lower,upper=upper)
            
        dp_1=(((theta[:,2].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/((theta[:,4].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,4].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,2].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,0].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]   
        dp_2=(((theta[:,2].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,2].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,0]).reshape((-1,1))**2)**0.5).reshape((1,-1))[0]
        dp_3=(((theta[:,7].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/((theta[:,8].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*(theta[:,8].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*(theta[:,7].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,9].reshape((-1,1))*theta[:,5].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]   
        dp_4=(((theta[:,7].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*(theta[:,7].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,5].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]
            
        dp=pm.math.stack([dp_1,dp_2,dp_3,dp_4],axis=1)
        dp=pm.math.switch(dp>4.5, 4.5, dp)
        pc = 0.25714*dp+0.07074*dp**2-0.02046*dp**3-0.0024*dp**4-0.00632*dp**5+0.00503*dp**6-0.00146*dp**7+0.00021*dp**8-0.000015*dp**9+0.00000041*dp**10+0.25
        nCorrect_obs=pm.Binomial('nCorrect',n=nTrials, p=pc, observed=nCorrect)


    #view the model structure
    graph=pm.model_to_graphviz(HBM)
    graph.render(filename='results/model structure/model strucutre_HBM', format='pdf')

    #variational inference begin
    with HBM:
        #Variational inference
        #approx=pm.fit(n=nIterations,method='advi') #iteration
        #idata=approx.sample(nDraws) #draw


        #Metropolis Hasting
        idata = pm.sample(draws=10000, chains=4, tune=10000)

    #summary
    summary=az.summary(idata, round_to=4, hdi_prob=0.95)
    summary.to_csv('results/summary/summary_HBM.csv', index=True)
    
    #save to a json file
    idata.to_json(filename='results/inference data/idata_HBM.json')

    print('all down!')


if __name__ == '__main__':
    main()