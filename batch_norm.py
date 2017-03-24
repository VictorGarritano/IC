import numpy as np

eps = 1e-8 #For numerical stability

#Forward pass

mu = (1.0/N) * np.sum(h, axis=0) # Size (H,) mini-batch mean for each hidden node
sigma2 = (1.0/N) * np.sum(np.square(h - mu), axis=0) #Size (H,) mini-batch variance for each hidden node
h_hat = (h-mu) / np.sqrt(sigma2 + eps) # Size (N,H) normalized layer output (zero mean and unit variance)
y = gamma * h_hat + beta # h_hat squashed through a linear function with parameters gamma and beta 

#Backward pass

dbeta = np.sum(dy, axis=0)
dgamma = np.sum(((h-mu) / np.sqrt(sigma2 + eps)) * dy, axis=0)
dh = (1.0/N) * (gamma/np.sqrt(sigma2 + eps)) * (N * dy - np.sum(dy, axis=0)
    - ((h - mu)/(var + eps)) * np.sum(dy * (h - mu), axis=0))
