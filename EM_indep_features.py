import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv('./Dados-medicos.csv', sep = ' ',
                   names = ['age', 'weight', 'carga-final', 'max-vo2'])
ntot = len(data)
data.head(n =5)

from random import randint

class EM :
    
    def random_initialisation(self):
        self.pi = np.array([1/self.k])
        self.data_for_k = np.array([[]])
        
        muk = np.array([])
        vark = np.array([])
            
        for jbis in range(self.m):
                muk = np.append(muk, randint(0, 15))
                vark = np.append(vark, np.random.random(1))
        
        self.mu = np.array(np.array([muk]))
        self.var = np.array(np.array([vark]))
        
        for c in range(self.k - 1):
            self.pi = np.append(self.pi, 1/self.k)
            self.data_for_k = np.concatenate((self.data_for_k, np.array([[]])))
            muk = np.array([])
            vark = np.array([])
            # for all features, different var and mean
            for j in range(self.m):
                muk = np.append(muk, randint(0, 15))
                vark = np.append(vark, np.array([np.random.random(1)]))
            self.mu = np.concatenate((self.mu, np.array([muk])))
            self.var = np.concatenate((self.var, np.array([vark])))
            
            #self.var = np.concatenate((self.var, np.array([vark])))
    
    def estimate_new_var_means_k(self, k):
        muk = np.array([])
        vark = np.array([])
        cluster_k = self.data_for_k[k]

        # for all features, different means and vars
        for j in range(m):
            muk = np.append(muk, np.mean(cluster_k[j]))
            vark = np.append(vark, np.var(cluster_k[j]))
        return muk, vark  
    
    def E_step(self, X):
        for i in range(self.n):
            posteriors = []
            for c in range(self.k):
                likelyhood_k = 1
                # calculate the likelihoods of all the joint independant features conditionally to class k
                for j in range(self.m):
                    likelyhood_k *= stats.norm.pdf(X.iloc[i,j], self.mu[c][j], mt.pow(self.var[c][j], 1/2))
                # multiplicating the prior
                posterior_k = likelyhood_k * self.pi[c]
                posteriors = np.append(posteriors, posterior_k)
            # get the MAP of all the posterior 
            print("posteriors \n" + str(posteriors))
            map_index = np.argmax(posteriors)
            # fill the temporary corresponding cluster k (MAP) with data i
            self.data_for_k[map_index] = np.concatenate((np.array([self.data_for_k[map_index]]), X.iloc[i,:]))
    
    def M_step(self):
        for c in range(self.k):
            self.pi[c] = len(self.data_for_k[c])/self.n
            self.mu[c], self.var[c] = self.estimate_new_var_means_k(k)
    
    def run(self, X, k, iter_max):
        # number of features x
        self.m = np.shape(X)[1]
        # number of data Xi
        self.n = np.shape(X)[0]
        # number of class c
        self.k = k
        # random initialisation
        self.random_initialisation()
        
        print("Random initialisation done \n" +
             " means \n" + str(self.mu) + "\n" +
             " vars \n" + str(self.var) + "\n" +
             " pis \n" + str(self.pi))  
        
        for i in range(iter_max) :
            self.E_step(X)
            self.M_step()

        return self.mu, self.var, self.pi