
# coding: utf-8

# ### The Viterbi Algorithm for Hidden Markov Models (HMM)
# 
# Hidden Markov Models are a way of modeling systems that have some kind "hidden" stateful process behind them. They are based on the Markov assumption:
# 
# $P(s_t|s_1,s_2,...,s_{t-1})=P(s_t|s_{t-1})$
# 
# Viterbi allows us to determine, given some set of observations ${y_1,y_2,...,y_T}$, the most likely sequence of states that generated those outputs (also called "emissions").
# 
# We define our HMM as follows:
# 
# Let $O = {o_1,o_2,...,o_N}$ be the *observational alphabet*, or the finite set of all possible observations.
# 
# Let $S = {s_1,s_2,...,s_K}$ be the set of possible states.
# 
# Let $A = a_{ij}$ where $a_{ij}$ is the probability of transitioning from state $i$ to state $j$ where $i,j\in \{1,...,K\}$
# 
# Let $B = b_{ij}$ where $b_{ij}$ is the probability of state $i$ "emitting" observation $j$ where $i\in\{1,...,K\}$ and $j\in\{1,...N\}$
# 
# Let $\Pi = \{\pi_1,\pi_2,...,\pi_K\}$ where $\pi_i$ is the prior probability of state $i$.
# 
# *Note: We omit the formal specification of $O$ in the implementation below since it is not needed for the algorithm. We assume that all values included in the observation sequence $Y\in\{1,2,...,T\}$, or more specifically,$Y\in\{0,1,...,T-1\}$ when taking Python zero indexing into account.*
# 
# So if we want to compute the probability of some sequence of observations $Y$ given some HMM $\lambda$, we can do so as follows:
# 
# $P(Y|\lambda) = \sum_S P(S|\lambda)P(Y|S,\lambda)$
# 
# The algorithm for performing this computation has three main parts:
# 
# (1) Initialization
# 
# Set all of the emission probabilities at $t_0$ to their priors.
# 
# For every $i$, set $V_{i,1} = \pi_ib_{i,y_1}$ and $Q_{i,1} = 0$
# 
# (2) Induction
# 
# For every $i$ from $2$ to $T$ and every $j$ from $1$ to $N$, find the most likely path so far up to $j$:
# 
# $V_{j,i} = max_{1\le k\le K}(V_{k,i-1}a_{ij}b_{j,y_i})$
# 
# $Q_{j,i} = argmax_{1\le k\le K}(V_{k,i-1}a_{ij})$
# 
# (3) Backtracking
# 
# Reconstruct the best possible state sequence computed by the induction step.
# 
# $Z_T = argmax_{1\le k\le K}(V_{k,T})$
# 
# For $t = T -1,T - 2,...,1$ : $Z_{t-1} = V_{Z_t,t}$ and $S*_{t-1} = S_{Z_{t-1}}$
# 
# *Note: In the implementation below, $T_1$ is $V$ and $T_2$ is $Q$*

# In[1]:


import numpy as np


# In[11]:


class HMM:
    def __init__(self, states, priors, transition_mat, emission_mat):
        self.states = states
        self.priors = priors
        self.transition_mat = transition_mat
        self.emission_mat = emission_mat
    
    # Implementation of Viterbi algorithm for simple HMM.
    # https://en.wikipedia.org/wiki/Viterbi_algorithm
    def viterbi(self, observations):
        # Step 0. Definitions
        # K is our number of possible states
        K = len(self.states)
        # S is our state space of length K
        S = np.array(self.states)
        # A is our transition probability table with dims KxK
        A = np.array(self.transition_mat)
        # B is our emissions probability table with dims KxN,
        # where N is the number of possible observations
        B = np.array(self.emission_mat)
        # Pi is our array of priors for the initial state of length K
        Pi = np.array(self.priors)
        # Y is our sequence of observations y1,y2,...,yt
        Y = np.array(observations)
        T = len(Y)
        # Index array for the best-path
        Z = np.zeros(T, dtype=int)
        # Best path, (x1,x2,...,xt)
        X = [S[0]]*T
        # Create our lookup table for the probability of the most likely path thus far:
        # V_ij = (x1,x2,...,xj) with xj=si that generates our observations (y1,y2,...,yj)
        V = np.zeros((K, T), dtype=float)
        # Create our lookup table for the previous state of the most likely path thus far:
        # Q_ij = (x1,x2,...,xj-1)
        Q = np.zeros((K, T), dtype=int)
        # Perform some assertions to make sure our parameters match specification.
        assert A.shape == (K,K)
        assert B.shape[0] == K
        assert len(Pi) == K
        assert T <= B.shape[1]
        
        # Step 1. Initialization
        for i in xrange(K):
            V[i,0] = Pi[i]*B[i,Y[0]]
        
        # Step 2. Induction
        # For each state and observation, compute the most probable path at state j and observation i.
        for (i, y) in enumerate(Y):
            # Skip first iteration for Y; we want to start at second index.
            # y2,y3,...,yT
            if i == 0:
                continue
            for j in xrange(K):
                p = [V[k,i-1]*A[k,j]*B[j,y] for k in xrange(K)]
                Q[j,i] = np.argmax(p)
                V[j,i] = p[Q[j,i]]
                
        # Step 3. Backtracking
        # Rebuild most probable sequence of states.
        Z[-1] = np.argmax(V[:,-1])
        X[-1] = S[Z[-1]]
        # Iterate descending from Y_len-1 (inclusive) to 0 (exclusive)
        for i in xrange(T - 1, 0, -1):
            Z[i-1] = Q[Z[i],i]
            X[i-1] = S[Z[i-1]]
        return X


# In[13]:


# Test on our hypothetical stock-prices example.
states = [1, 2, 3]
t_mat = np.array([[0.6,0.2,0.2],[0.5,0.3,0.2],[0.4,0.1,0.5]])
e_mat = np.array([[0.7,0.1,0.2],[0.1,0.6,0.3],[0.3,0.3,0.4]])
priors = [0.5,0.2,0.3]
hmm = HMM(states, priors, t_mat, e_mat)

print hmm.viterbi([0,0,0])

from itertools import permutations
Y_all = []
for p in permutations(xrange(len(states))):
    Y_all.append(p)
for Y in Y_all:
    print '{0} -> {1}'.format(Y, hmm.viterbi(Y))

