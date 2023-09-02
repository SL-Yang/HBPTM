# HBPTM pack for predicting subjects' behavioral response in attention tasks.

The program is designed to fit the model described in the perceptual template model (PTM), see original paper External noise distinguishes attention mechanisms (https://www.sciencedirect.com/science/article/pii/S0042698997002733)

The simulation primarily uses the PYMC library in Python and the figures are made by matplotlib.pyplot

compute_BIP.py models individual subject responses using Bayesian Inference Procedure (BIP),compute_HBM.py models a group of subjects' response by setting hyper-parameters regulating sub-parameters using the Hierarchical Bayesian Model (HBM),model_comparation.py compares BIP and HBM model's performance,analyze.py plot figures about simulation results, including model structure, parameters' posterior distribution, and the trade-off between parameters

The behavioral data and results are not published but you are welcome to use the program to fit your own data. The data needed includes noise, contrast, trials per block (under different conditions), 
number of correct per block (under different conditions).
