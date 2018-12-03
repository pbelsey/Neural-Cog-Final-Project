
from __future__ import division 
import numpy as np 
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
from scipy.stats import sem
import seaborn as sns
import string
import matplotlib.pyplot as plt

def update_Qi(Qval, reward, alpha, gamma, beta):
    """ update q-value of selected action, given reward and alpha
    """
    return Qval + alpha * (reward - Qval)

class Player(object):
	# defines other players in BeerGame 
	def __init__(self, demand=0, backorders=0, inventory=4, incoming1=4, incoming2=4): 
		self.backorders = backorders 
		self.inventory = inventory
		self.demand = demand
		self.incoming1 = incoming1 
		self.incoming2 = incoming2 

	def get_state(self): 
		return (inventory, backorders)

	def fill_orders(self, otherplayer): 
		if self.inventory >= orders: 
			self.inventory -= orders 
		else if self.inventory = 0 
			self.backorders += self.orders
		else: 
			self.backorders = orders - self.inventory 
			self.inventory = 0 

	def place_orders(self): 
		if demand < 0: 
			raise Exception("you placed an order for the td agent")
		return self.demand 

	def receive_cases(self):
		self.inventory += self.incoming2
		self.incoming2 = self.incoming1 


class BeerGame(object):
	# defines the beer game task 

	# REQUIRES: lowBound <= demand <= upperBound
		# lowBound (int): lowest possible no. of cases one can order
		# highBound (int): highest possible no. of cases one can order 
		# demand (int): external customer demand 
	# EXPECTED: 
		# span = array([lowbound, ..., highBound])
		# target = target 
	def __init__(self, lowBound, highBound, inventory, demand):
		self.target = target 
		self.actions = np.arange(lowBound, highBound+1)
		self.factory = Player(demand)
		self.distributer = Player() # td-learning agent
		self.wholesaler = Player(demand)
		self.retailer = Player(demand)
		self.orders = [4,0,4,4] # holds orders from last trial
		self.players = [self.factory, self.distributer, self.wholesaler, self.retailer]
		self.no_players = 4

		if (demand < lowBound or demand > highBound):
			raise Exception('target not included in the indicated actions')

	# REQUIRES: 
	def get_state(self):
		self.distributer.get_state()

	def get_reward(self, cases_ordered): 
		n = len(self.players)
		for i in n: 


"""
class MultiArmedBandit(object):
    """ defines a multi-armed bandit task
    ::Arguments::
        preward (list): 1xN vector of reward probaiblities for each of N bandits
        rvalues (list): 1xN vector of payout values for each of N bandits
    """
    def __init__(self, preward=[.9, .8, .7], rvalues=[1, 1, 1]):
        self.preward = preward
        self.rvalues = rvalues
        try:
            assert(len(self.rvalues)==len(self.preward))
        except AssertionError:
            self.rvalues = np.ones(len(self.preward))

    def set_params(self, **kwargs):
        error_msg = """preward and rvalues must be same size
                    setting all rvalues to 1"""
        kw_keys = list(kwargs)
        if 'preward' in kw_keys:
            self.preward = kwargs['preward']
            if 'rvalues' not in kw_keys:
                try:
                    assert(len(self.rvalues)==len(self.preward))
                except AssertionError:
                    self.rvalues = np.ones(len(self.preward))

        if 'rvalues' in kw_keys:
            self.rvalues = kwargs['rvalues']
            try:
                assert(len(self.rvalues)==len(self.preward))
            except AssertionError:
                raise(AssertionError, error_msg)

    def get_feedback(self, action_ix):
        pOutcomes = np.array([self.preward[action_ix], 1-self.preward[action_ix]])
        Outcomes = np.array([self.rvalues[action_ix], 0])
        feedback = np.random.choice(Outcomes, p=pOutcomes)
        return feedback
"""

# adapted from qlearning.py in ADMCode 
class TDagent(object):
    """ defines the learning parameters of single td-learning-learning agent
    in a 
    ::Arguments::
        alpha (float): learning rate
        beta (float): inverse temperature parameter
        gamma (float): discount rate 
        preward (list): 1xN vector of reward probaiblities for each of N bandits
        rvalues (list): 1xN vector of payout values for each of N bandits
                        IF rvalues is None, all values set to 1
    """
    def __init__(self, alpha=.04, beta=3.5, gamma=.02, epsilon=.1, lowbound=0, 
    	highbound=60, inventory=4, demand=4):
        if (inventory<0 or demand<=0 or lowbound<0 or highbound<lowbound):
            raise Exception('parameters make no sense')
        self.beer = BeerGame(lowbound=lowbound, highbound=highbound, inventory=inventory, demand=demand)
        self.updateQ = lambda Qval, r, alpha: Qval + alpha*(r - Qval)
        self.updateP = lambda Qvector, act_i, beta: np.exp(beta*Qvector[act_i])/np.sum(np.exp(beta*Qvector))
        self.set_params(alpha=alpha, beta=beta, gamma=gamma, epsilon=epsilon)

    def set_params(self, **kwargs):
        """ update learning rate, inv. temperature, and/or
        epsilon parameters of q-learning agent
        """

        kw_keys = list(kwargs)

        if 'alpha' in kw_keys:
            self.alpha = kwargs['alpha']

        if 'beta' in kw_keys:
            self.beta = kwargs['beta']

        if 'gamma' in kw_keys: 
        	self.gamma = kwargs['gamma']

        if 'epsilon' in kw_keys:
            self.epsilon = kwargs['epsilon']

        self.nact = self.highbound-self.lowbound
        self.actions = np.arange(self.nact)

	def play_beergame(self, ntrials=1000, get_output=True):
	    """ simulates agent performance on a beer game task 
	    ::Arguments::
	        ntrials (int): number of trials to play bandits
	        get_output (bool): returns output DF if True (default)
	    ::Returns::
	        DataFrame (Ntrials x Nbandits) with trialwise Q and P
	        values for each bandit
	    """
	    pdata = np.zeros((ntrials+1, self.nact))
	    pdata[0, :] = np.array([1/self.nact]*self.nact)
	    qdata = np.zeros_like(pdata)
	    self.choices = []
	    self.feedback = []

	    for t in range(ntrials):

	        # select bandit arm (action)
	        act_i = np.random.choice(self.actions, p=pdata[t, :])

	        # get reward and state  
	        (r,s) = self.bandits.get_feedback(act_i)

	        # update value of selected action
	        qdata[t+1, act_i] = update_Qi(qdata[t, act_i], r, self.alpha)

	        # broadcast old q-values for unchosen actions
	        for act_j in self.actions[np.where(self.actions!=act_i)]:
	            qdata[t+1, act_j] = qdata[t, act_j]

	        # update action selection probabilities and store data
	        pdata[t+1, :] = update_Pall(qdata[t+1, :], self.beta)
	        self.choices.append(act_i)
	        self.feedback.append(r)

	    self.pdata = pdata[1:, :]
	    self.qdata = qdata[1:, :]
	    self.make_output_df()

	    if get_output:
	        return self.data.copy()









