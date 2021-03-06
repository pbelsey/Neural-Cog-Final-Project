
from __future__ import division 
import numpy as np 
import random
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
from scipy.stats import sem
import seaborn as sns
import string
import matplotlib.pyplot as plt

def update_Qi(Qval0, Qval1, reward1, alpha, gamma):
    """ update q-value of selected action, given reward and alpha
    """
    return Qval0 + alpha * (reward1 + gamma*Qval1 - Qval0)

def update_Pall(Qvector, beta):
    """ update vector of action selection probabilities given
    associated q-values
    """
    return np.array([np.exp(beta*Q_i) / np.sum(np.exp(beta * Qvector)) for Q_i in Qvector])

class Player(object):
    # defines other players in BeerGame 
    def __init__(self, demand=0, backorders=0, inventory=4, incoming1=4, incoming2=4): 
        self.backorders = backorders 
        self.inventory = inventory
        self.demand = demand
        self.incoming1 = incoming1 
        self.incoming2 = incoming2 

    def get_state(self): 
        return (self.inventory, self.backorders)

    def fill_orders(self, nextplayer, orders): 
        if self.inventory >= orders and self.backorders == 0: 
            self.inventory -= orders
            nextplayer.receive_cases(orders) 
        elif self.inventory >= (orders + self.backorders): 
            nextplayer.receive_cases(orders + self.backorders)
            self.inventory -= (orders + self.backorders)
            self.backorders = 0 
        elif self.inventory >= self.backorders and self.inventory < (orders + self.backorders): 
            nextplayer.receive_cases(self.backorders + orders - self.inventory) 
            self.inventory -= self.backorders
            self.backorders = orders - self.inventory
            self.inventory = 0 
        elif self.inventory > 0 and self.inventory < self.backorders:
            nextplayer.receive_cases(self.backorders - self.inventory) 
            self.backorders -= self.inventory 
            self.backorders += orders 
            self.inventory = 0 
        else: 
            nextplayer.receive_cases(0)
            self.backorders += orders 

    def receive_cases(self, cases):
        self.inventory += self.incoming2
        self.incoming2 = self.incoming1 
        self.incoming1 = cases 

class BeerGame(object):
    # defines the beer game task 

    # REQUIRES: lowBound <= demand <= upperBound
        # lowBound (int): lowest possible no. of cases one can order
        # highBound (int): highest possible no. of cases one can order 
    # demand (int): external customer demand 
    # EXPECTED: 
        # span = array([lowbound, ..., highBound])
        # target = target 
    def __init__(self, lowBound=0, highBound=10, inventory=4, demand=4):
        self.actions = np.arange(lowBound, highBound+1)
        self.factory = Player(demand,0,inventory)
        self.distributer = Player(demand,0,inventory) # td-learning agent
        self.wholesaler = Player(demand,0,inventory)
        self.retailer = Player(demand,0,inventory)
        self.customer = Player(demand,0,inventory)
        self.orders = [demand,self.actions[random.randrange(0,len(self.actions))],demand,demand,demand] # holds orders from last trial
        self.players = [self.factory, self.distributer, self.wholesaler, self.retailer, self.customer]

        if (demand < lowBound or demand > highBound):
            raise Exception('target not included in the indicated actions')

    # REQUIRES: 
    def get_state(self):
        self.distributer.get_state()

    def get_reward(self, cases_ordered): 
        n = len(self.players)
        for i in range(n-1):
            self.players[i].fill_orders(self.players[i+1], self.orders[i])
        orders = cases_ordered
        (inventory, backorders) = self.distributer.get_state()
        return (inventory * -0.5 + backorders * -0.5)


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
    def __init__(self, alpha=.1, beta=3.5, gamma=.99, epsilon=.1, lowbound=0, 
    	highbound=10, inventory=5, demand=5):
        if (inventory<0 or demand<=0 or lowbound<0 or highbound<lowbound):
            raise Exception('parameters make no sense')
        self.beergame = BeerGame(lowBound=lowbound, highBound=highbound, inventory=inventory, demand=demand)
        self.highbound = highbound
        self.lowbound = lowbound
        self.demand = demand
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

            # select bandit arm (action) from state space 
            act_i = np.random.choice(self.actions, p=pdata[t, :])
            
            # get reward for current action  
            r = self.beergame.get_reward(act_i)
            
            if t>0: 
            # update value of selected action
                qdata[t+1, act_i] = update_Qi(qdata[t-1, self.last], qdata[t, act_i], r, self.alpha, self.gamma)
                        
            # broadcast old q-values for unchosen actions
            for act_j in range(self.nact):
                if act_j == act_i: continue 
                qdata[t+1, act_j] = qdata[t, act_j]
                
            self.last = act_i
            # update action selection probabilities and store data
            pdata[t+1, :] = update_Pall(qdata[t+1, :], self.beta)
            self.choices.append(act_i)
            self.feedback.append(r)

        self.pdata = pdata[1:, :]
        self.qdata = qdata[1:, :]
        self.make_output_df()

        if get_output:
            return self.data.copy()

    def make_output_df(self):
        """ generate output dataframe with trialwise Q and P measures for each bandit,
        as well as choice selection, and feedback
        """
        df = pd.concat([pd.DataFrame(dat) for dat in [self.qdata, self.pdata]], axis=1)
        columns = np.hstack(([['{}{}'.format(x, c) for c in self.actions] for x in ['q', 'p']]))
        df.columns = columns
        df.insert(0, 'trial', np.arange(1, df.shape[0]+1))
        df['choice'] = self.choices
        df['feedback'] = self.feedback
#            r = np.array(self.bandits.rvalues)
#            p = np.array(self.bandits.preward)
        df['optimal'] = self.demand
        df.insert(0, 'agent', 1)
        self.data = df.copy()






