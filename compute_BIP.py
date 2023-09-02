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
    iSubject=1
    nTests=1
    nDraws=100000
    nIterations=100000

    #read data
    my_data=pd.read_csv('data/data40q_var.csv')
    #contrast level , noise level, behaviral correct response
    c=tt.constant(my_data['Contrast'][72*(iSubject-1):72*(iSubject)])
    n=tt.constant(my_data['Noise'][72*(iSubject-1):72*(iSubject)])
    nCorrect=my_data[['nCorrect1','nCorrect2','nCorrect3','nCorrect4']][72*(iSubject-1):72*(iSubject)]

    #uniform distribution lower and upper bound for 10 parameters
    lower=tt.constant([0,0,0,0,0,0,0,0,0,0])
    upper=tt.constant([1,1,10,3,2,1,1,10,2,2])

    #gamma distribution parameter alpha and beta with shape 1*10 population level
    sig_a=tt.ones(10)*5000
    sig_b=tt.ones(10)*3

    BIP=pm.Model()

    with BIP:
        mu=pm.Uniform('mu', lower=lower, upper=upper)
        sigma=pm.Gamma('sigma', alpha=sig_a, beta=sig_b)
        theta=pm.TruncatedNormal('theta', mu=mu, tau=sigma, lower=lower, upper=upper)
       
        #compute discriminability
        dp_1=(((theta[2]*c)**theta[3])/((theta[4]*n)**(theta[3]*2)+theta[1]**2*(theta[4]*n)**(theta[3]*2)+theta[1]**2*(theta[2]*c)**(theta[3]*2)+(theta[0])**2)**0.5) 
        dp_2=(((theta[2]*c)**theta[3])/(n**(theta[3]*2)+theta[1]**2*n**(theta[3]*2)+theta[1]**2*(theta[2]*c)**(theta[3]*2)+(theta[0])**2)**0.5)  
        dp_3=(((theta[7]*c)**theta[3])/((theta[8]*n)**(theta[3]*2)+theta[6]**2*(theta[8]*n)**(theta[3]*2)+theta[6]**2*(theta[7]*c)**(theta[3]*2)+(theta[9]*theta[5])**2)**0.5)      
        dp_4=(((theta[7]*c)**theta[3])/(n**(theta[3]*2)+theta[6]**2*n**(theta[3]*2)+theta[6]**2*(theta[7]*c)**(theta[3]*2)+(theta[5])**2)**0.5)
        
        dp=pm.math.stack([dp_1,dp_2,dp_3,dp_4],axis=1)
        dp=pm.math.switch(dp>4.5, 4.5, dp)
        
        #convert discriminbility to percentage correct using a polynomial from linear regression
        pc = 0.25714*dp+0.07074*dp**2-0.02046*dp**3-0.0024*dp**4-0.00632*dp**5+0.00503*dp**6-0.00146*dp**7+0.00021*dp**8-0.000015*dp**9+0.00000041*dp**10+0.25

        #compare estimated number of correctness with observed data, find the best fitting parameters
        nCorrect_obs=pm.Binomial('nCorrect',n=nTrials, p=pc, observed=nCorrect)

    #view the model structure
    graph=pm.model_to_graphviz(BIP)
    graph.render(filename='results/model structure/model strucutre_BIP', format='pdf')

    #variational inference begin
    with BIP:
        #approx=pm.fit(n=nIterations,method='advi') #iteration
        #idata=approx.sample(nDraws) #draw
        idata = pm.sample(draws=10000, chains=4, tune=10000)
        
    summary=az.summary(idata, round_to=4, hdi_prob=0.95)
    summary.to_csv('results/summary/summary_BIP.csv', index=True)


    #save to a json file
    idata.to_json(filename='results/inference data/idata_BIP.json')

    print('all done!')

if __name__ == '__main__':
    main()