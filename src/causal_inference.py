import torch
from sklearn.linear_model import LogisticRegression
import numpy as np

def estimate_propensity_scores(X, A):
    model = LogisticRegression()
    model.fit(X, A)
    propensity_scores = model.predict_proba(X)[:, 1]
    return propensity_scores

def doubly_robust_estimation(X, A, Y, propensities, model_predictions):
    ips_weights = A / propensities + (1 - A) / (1 - propensities)
    pseudo_outcomes = A * (Y - model_predictions) / propensities + model_predictions
    dr_estimate = np.mean(pseudo_outcomes)
    return dr_estimate
