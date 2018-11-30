
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

class Player(object):
	# defines other players in BeerGame 

	def __init__(self, demand=-1, backorders=0, inventory=0): 
		self.backorders = backorders 
		self.inventory = inventory
		self.demand = demand

	def fillOrders(self, orders): 
		if self.inventory >= orders: 
			self.inventory -= orders 
		else if self.inventory = 0 
			self.backorders += orders 
		else: 
			self.backorders = orders - self.inventory 
			self.inventory = 0 

	def placeOrders(self): 
		if demand < 0: 
			raise Exception("you placed an order for the td agent")
		return self.demand 

	def receiveOrders(self, orders):
		self.inventory += orders


class BeerGame(object):
	# defines the beer game task 

	# REQUIRES: lowBound <= demand <= upperBound
		# lowBound (int): lowest possible no. of cases one can order
		# highBound (int): highest possible no. of cases one can order 
		# demand (int): external customer demand 
	# EXPECTED: 
		# span = array([lowbound, ..., highBound])
		# target = target 
	def __init__(self, lowBound=2, highBound=8, inventory= ,demand=4):
		self.target = target 
		self.actions = np.arange(lowBound, highBound+1)
		self.factory = Player(demand)
		self.distributer = Player() # td-learning agent
		self.wholesaler = Player(demand)
		self.retailer = Player(demand)

		if (demand < lowBound or demand > highBound):
			raise Exception('target not included in the indicated actions')

	# REQUIRES: 
	def get_feedback(self, action_ix, week): 
		









