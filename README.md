# FunctionalTreatment
Code to reproduce the simulation results of the FSW, OR, DR and FLR in Tan R., Zhang Z., Huang W. and Yin G. (2025). Causal effect of functional treatment. Journal of Machine Learning Research, 26, 1--39.

Select a model using `model_opt` and run Simulation.m to reproduce the simulation results in the paper. Note that `n_rep = 200` may take a long time to run.

weight_con_LOO.m computes the weight pi;

FLR_**.m's compute the functional-linear-regression-related models;

CV_**.m's perform the CV selection on tuning parameters;

FPCA.m performs the functional principal component analysis.
