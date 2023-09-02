import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as tt
import graphviz


#number of trials, subjects, test
nTrials=40
nSubjects=18
nTests=1
nDraws=100000
nIterations=150000
nContrast=9
nNoise=8

#read data
my_data=pd.read_csv('data/data40q_var.csv')
#contrast level , noise level, behaviral correct response
c=tt.constant(my_data['Contrast']).reshape((nSubjects,nContrast*nNoise))
n=tt.constant(my_data['Noise']).reshape((nSubjects,nContrast*nNoise))
nCorrect=my_data[['nCorrect1','nCorrect2','nCorrect3','nCorrect4']]

lower_on=tt.constant([0,0,0,0,0,0,0,0,0,0])
upper_on=tt.constant([1,1,10,3,2,1,1,10,2,2])

lower_off=tt.constant([0,0,0,0,0,0,0])
upper_off=tt.constant([1,1,10,3,1,1,10])

attention_on=pm.Model()

with attention_on:

    mu=pm.Uniform('mu', lower=lower_on, upper=upper_on)
        
    sigma=pm.Gamma('sigma', alpha=tt.ones(10)*5000, beta=tt.ones(10)*3)
    pho=pm.TruncatedNormal('pho', mu=tt.tile(mu,(nSubjects,1)), tau=sigma, lower=lower_on,upper=upper_on)
        
    delta=pm.Gamma('delta', alpha=tt.ones(10)*5000, beta=tt.ones(10)*3)
    theta=pm.TruncatedNormal('theta', mu=tt.tile(pho,(nTests,1)), tau=delta, lower=lower_on,upper=upper_on)
    
    dp_1=(((theta[:,2].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/((theta[:,4].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,4].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,2].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,0].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]   
    dp_2=(((theta[:,2].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,2].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,0]).reshape((-1,1))**2)**0.5).reshape((1,-1))[0]
    dp_3=(((theta[:,7].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/((theta[:,8].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*(theta[:,8].reshape((-1,1))*n)**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*(theta[:,7].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,9].reshape((-1,1))*theta[:,5].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]   
    dp_4=(((theta[:,7].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,6].reshape((-1,1))**2*(theta[:,7].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,5].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]
        
            
    dp=pm.math.stack([dp_1,dp_2,dp_3,dp_4],axis=1)
    dp=pm.math.switch(dp>4.5, 4.5, dp)
    pc = 0.25714*dp+0.07074*dp**2-0.02046*dp**3-0.0024*dp**4-0.00632*dp**5+0.00503*dp**6-0.00146*dp**7+0.00021*dp**8-0.000015*dp**9+0.00000041*dp**10+0.25
    nCorrect_obs=pm.Binomial('nCorrect',n=nTrials, p=pc, observed=nCorrect)

with attention_on:
    approx_on=pm.fit(n=nIterations,method='advi') #iteration
    idata_on=approx_on.sample(nDraws) #draw
    L_on=pm.find_MAP(return_raw=True)
    pm.compute_log_likelihood(idata_on)

summary_on=az.summary(idata_on)
num_para_on=summary_on.shape[0]
logp_on=-L_on[1]['fun']
BPIC_on=-2 * logp_on + num_para_on * np.log(nDraws)


attention_off=pm.Model()

with attention_off:

    mu=pm.Uniform('mu', lower=lower_off, upper=upper_off)
        
    sigma=pm.Gamma('sigma', alpha=tt.ones(7)*5000, beta=tt.ones(7)*3)
    pho=pm.TruncatedNormal('pho', mu=tt.tile(mu,(nSubjects,1)), tau=sigma, lower=lower_off,upper=upper_off)
        
    delta=pm.Gamma('delta', alpha=tt.ones(7)*5000, beta=tt.ones(7)*3)
    theta=pm.TruncatedNormal('theta', mu=tt.tile(pho,(nTests,1)), tau=delta, lower=lower_off,upper=upper_off)
            
    dp_1=(((theta[:,2].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,2].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,0].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]   
    dp_2=(((theta[:,2].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,1].reshape((-1,1))**2*(theta[:,2].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,0]).reshape((-1,1))**2)**0.5).reshape((1,-1))[0]
    dp_3=(((theta[:,6].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,5].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,5].reshape((-1,1))**2*(theta[:,6].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,4].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]   
    dp_4=(((theta[:,6].reshape((-1,1))*c)**theta[:,3].reshape((-1,1)))/(n**(theta[:,3].reshape((-1,1))*2)+theta[:,5].reshape((-1,1))**2*n**(theta[:,3].reshape((-1,1))*2)+theta[:,5].reshape((-1,1))**2*(theta[:,6].reshape((-1,1))*c)**(theta[:,3].reshape((-1,1))*2)+(theta[:,4].reshape((-1,1)))**2)**0.5).reshape((1,-1))[0]
    
    dp=pm.math.stack([dp_1,dp_2,dp_3,dp_4],axis=1)
    dp=pm.math.switch(dp>4.5, 4.5, dp)
    pc = 0.25714*dp+0.07074*dp**2-0.02046*dp**3-0.0024*dp**4-0.00632*dp**5+0.00503*dp**6-0.00146*dp**7+0.00021*dp**8-0.000015*dp**9+0.00000041*dp**10+0.25
    nCorrect_obs=pm.Binomial('nCorrect',n=nTrials, p=pc, observed=nCorrect)

with attention_off:
    approx_off=pm.fit(n=nIterations,method='advi') #iteration
    idata_off=approx_off.sample(nDraws) #draw
    L_off=pm.find_MAP(return_raw=True)
    pm.compute_log_likelihood(idata_off)

summary_off=az.summary(idata_off)
num_para_off=summary_off.shape[0]
logp_off=-L_off[1]['fun']
BPIC_off=-2 * logp_off + num_para_off * np.log(nDraws)


df_comp_loo = az.compare({"att": idata_on, "no-att": idata_off})
df_comp_loo.to_csv('results/comparation/att vs no-att.csv', index=True)

az.plot_compare(df_comp_loo, insample_dev=False, figsize=(12, 6))
plt.savefig("results/comparation/att vs no-att.png")


print('BPIC_on:', BPIC_on)
print('BPIC_off:', BPIC_off) 