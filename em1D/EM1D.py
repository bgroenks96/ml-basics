
# coding: utf-8

# ### 1-Dimensional Expectation Maximization

# In[1]:


import numpy as np
import scipy.stats as stats


# In[2]:


def em_1D(data, num_classes, iterations=1000):
    mus = np.linspace(min(data), max(data), num_classes)
    sds = np.random.uniform(low=1, high=10, size=num_classes)
    class_weights = np.random.uniform(low=-1.0, high=1.0, size=num_classes)
    log_l_hist = []
    for i in xrange(iterations):
        # Calculate weighted densities for all classes
        dvals = np.array([stats.norm.pdf(data, mus[i], sds[i])*class_weights[i] for i in xrange(num_classes)])
        # Sum along columns to get alpha values for each class
        alphas = np.sum(dvals, axis=0)
        # Get log likelihood from alphas
        log_l = np.sum(np.log(alphas))
        log_l_hist.append(log_l)
        
        alpha_mat = np.tile(alphas, (num_classes, 1))
        dprobs = dvals / alpha_mat
        ns = np.sum(dprobs, axis=1).T
        emus = np.dot(dprobs, data)
        mus = (emus / ns.T).T
        for (i,p) in enumerate(dprobs):
            v = np.sum(np.dot(p, (data - mus[i])**2)) / ns[i]
            sds[i] = np.sqrt(v)
        class_weights = ns / len(data)
        
        print 'mus: {0}'.format(mus)
        print 'sds: {0}'.format(sds)
        print 'weights: {0}'.format(class_weights)


# In[3]:


c1_data = np.random.normal(1, 2, 100)
c2_data = np.random.normal(2, 5, 100)
em_1D(np.concatenate((c1_data, c2_data)), 2)

