"""
This file generates pseudo-random data according to the distribution defined in the document
"Math/Confounders and data generation.pdf".
Author: Maximilian Janisch
"""

import pickle
from tqdm import tqdm

import numpy as np

from scipy import stats
from scipy.special import expit

import pandas as pd



def get_data_XTY_one_dimensional_with_hidden_confounding(n, a=-2, c=1/2, random_state=None, verbose=True, cor_hidden_visible=0.5):
    """
    Generates pseudo-random data distributed according to the distribution defined in section 2.1 of the document
    "Math/Confounders and data generation.pdf".
    :param n: Number of data points to generate.
    :param a: Mean of X.
    :param c: Shape parameter for Weibull distribution.
    :param random_state: Used to set the seed of numpy.random before generation of random numbers.
    :param verbose: If True will display a progress bar. If False it will not display a progress bar.
    :param: cor_hidden_visible denotes the desired Pearson correlation coefficient between the hidden and the visible
    :return: Pandas DataFrame with four columns (corresponding to X_{hidden}, X_{visible}, T and Y) and n rows (corresponding to the n
    generated pseudo-random samples).
    """
    np.random.seed(random_state)

    output = []
    iterator = tqdm(range(n)) if verbose else range(n)

    confounding_variance = 1 / (cor_hidden_visible**2) - 1  # I will explain this later
    for _ in iterator:
        X_hidden = stats.norm.rvs(loc=-2, scale=1)
        X_visible = X_hidden + stats.norm.rvs(loc=0, scale=confounding_variance**.5)
        T = stats.bernoulli.rvs(p=1/(1+np.exp(-X_hidden)))
        if T == 0:
            Z = stats.expon.rvs(scale=np.exp(-X_hidden))  # note: np.exp(-X_hidden) could be cached for more computational efficiency but would render the code less useful
        elif T == 1:
            Z = stats.weibull_min.rvs(c=c, scale=np.exp(-X_hidden))
        else:
            assert False
        output.append((X_hidden, X_visible, T, Z))

    return pd.DataFrame(output, columns=["Personal information (hidden)", "Personal information (visible)", "Treatment", "Time to event"])


data = get_data_XTY_one_dimensional_with_hidden_confounding(n=100, random_state=0)
print(data)

x=data["Personal information (hidden)"]
y=data["Time to event"]
t=data["Treatment"]


def data_loading_twin(train_rate = 0.8):
    """Load toy data.
  
  Args:
    - train_rate: the ratio of training data
    
  Returns:
    - train_x: features in training data
    - train_t: treatments in training data
    - train_y: observed outcomes in training data
    - train_potential_y: potential outcomes in training data
    - test_x: features in testing data
    - test_potential_y: potential outcomes in testing data      
  """
  
    # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
    ori_data = np.loadtxt("data/toydatawithconf.csv", delimiter=",",skiprows=1)

    # Define features
    no, dim = x.shape
        
    # Define potential outcomes
    potential_y = ori_data[:, 3:]
    # Die within 1 year = 1, otherwise = 0
    potential_y = np.array(potential_y < 9999,dtype=float)    
                
    ## Assign treatment
    coef = np.random.uniform(-0.01, 0.01, size = [dim,1])
    prob_temp = expit(np.matmul(x, coef) + np.random.normal(0,0.01, size = [no,1]))
        
    prob_t = prob_temp/(2*np.mean(prob_temp))
    prob_t[prob_t>1] = 1
        
    t = np.random.binomial(1,prob_t,[no,1])
    t = t.reshape([no,])

    ## Define observable outcomes
    y = np.zeros([no,1])
    y = np.transpose(t) * potential_y[:,1] + np.transpose(1-t) * potential_y[:,0]    
    y = np.reshape(np.transpose(y), [no, ])

    ## Train/test division
    idx = np.random.permutation(no)
    train_idx = idx[:int(train_rate * no)]
    test_idx = idx[int(train_rate * no):]
        
    train_x = x[train_idx,:]
    train_t = t[train_idx]
    train_y = y[train_idx]
    train_potential_y = potential_y[train_idx,:]

    test_x = x[test_idx,:]
    test_potential_y = potential_y[test_idx,:]
            
    return train_x, train_t, train_y, train_potential_y, test_x, test_potential_y