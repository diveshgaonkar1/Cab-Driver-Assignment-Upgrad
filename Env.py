import numpy as np
import math
import random
import itertools
from itertools import product

#Defining hyperparameters
m = 5   #Number of cities, ranges from 1 ..... m
t = 24  #Number of hours, ranges from 0 .... t-1
d = 7   #Number of days, ranges from 0 ... d-1
C = 5   #Per hour fuel and other costs
R = 9   #Per hour revenue from a passenger

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.locations = np.arange(0,5)            # Locations encoded as an integer 0 to 4
        self.hours = np.arange(0, 24,dtype=np.int) # Hours are encoded as 24
        self.days = np.arange(0, 7)                # days of week encoded as 0 to 6

        #Create action space of 20 possible commute actions
        self.action_space = [a for a in list(product(self.locations, self.locations)) if a[0] != a[1]]
        self.action_size = len(self.action_space)
        
        #Create state space where state is [location, hours, days]
        self.state_space = [a for a in list(product(self.locations, self.hours, self.days))]
        self.state_init =self.state_space[np.random.randint(len(self.state_space))]

        self.pickuptime = 0     #Placeholder for commute time when pickup & drop locations are not same
        self.droptime = 0       #Time spent in dropping passenger for single t
        self.TotalRideTime = 0  #Total commute time for pickup and drop. TotalRideTime=pickuptime+droptime
        self.state_size=36      #For architecture 1 state size will be m+t+d i.e 5+24+7
        
        self.reset()

    #Encoding state (or state-action) for NN input for architecture 1
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN.This method converts a given state into a vector format. Hint: 
           The vector is of size m + t + d."""
        loc = np.eye(m, dtype=np.int16)[int(state[0])]    #Location vector
        hour = np.eye(t, dtype=np.int16)[int(state[1])]   #Hour vector
        day = np.eye(d, dtype=np.int16)[int(state[2])]    #Day vector
        state_encod = np.concatenate((loc, hour, day))    #Combined state vector
        return state_encod

    #Obtaining number of requests at current state
    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        loc_index = state[0]
        if loc_index == 0 :
            requests = np.random.poisson(2)
        elif loc_index == 1 :
            requests = np.random.poisson(12)
        elif loc_index == 2 :
            requests = np.random.poisson(4)
        elif loc_index == 3 :
            requests = np.random.poisson(7)
        elif loc_index == 4 :
            requests = np.random.poisson(8)

        #Restrict max number of requests to 15
        if requests > 15:
            requests = 15

        #Fetch actions for number of requests received
        action_indices = random.sample(range(0, (m-1)*m), requests)
        actions = [self.action_space[i] for i in action_indices]
        
        #Add offline action to the set of possible actions
        actions.append((0, 0))  #Condition as (0,0) represents offline condition
        return  actions

    #Compute reward depending on ride completion
    def reward_func(self, state, action, Time_matrix,flag):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        #Reward function returns revenue earned from pickup point p to drop point q
        """ As per MDP definition
        𝑅(𝑠=𝑋𝑖𝑇𝑗𝐷𝑘) ={ 𝑅𝑘∗(𝑇𝑖𝑚𝑒(𝑝,𝑞)) − 𝐶𝑓 ∗(𝑇𝑖𝑚𝑒(𝑝,𝑞) + 𝑇𝑖𝑚𝑒(𝑖,𝑝))  𝑎=(𝑝,𝑞)
                                      −𝐶𝑓                            𝑎=(0,0)}"""
        if flag==1:
            reward=(self.droptime*R)-(C*(self.pickuptime*+self.droptime))
        else:
            reward=-self.pickuptime
        return reward
    
    #Compute next state basis current state and action
    def next_state_func(self, state, action, Time_matrix):
        
        """Takes state and action as input and returns next state"""

        if not isinstance(action,tuple):
            #Fetch current action
            action=self.action_space[action]
            
        #Initialise variables
        is_terminal=False
        self.pickuptime=0
        self.droptime=0

        currLocation=state[0]
        totalHours=state[1]
        totaldays=state[2]

        pickupLocation=action[0]
        dropLocation=action[1]

        #Compute next state basis action taken
        if action[0]!=0 and action[1]!=0:
            if currLocation!=pickupLocation:
                
                #Compute time taken to reach pickup point
                self.pickuptime=Time_matrix[currLocation][pickupLocation][int(totalHours)][totaldays]
                totalHours=self.pickuptime+totalHours

                #Update totalHours and totaldays
                totalHours,totaldays= self.DayModifier(totalHours,totaldays)
                
                #Set pickup location as current location
                currLocation=pickupLocation

            #Compute drop time
            self.droptime=Time_matrix[currLocation][dropLocation][int(totalHours)][totaldays]
            
            #Update totalHours and totaldays
            totalHours=totalHours+self.droptime
            totalHours,totaldays= self.DayModifier(totalHours,totaldays)
            
            #Compute total ride time
            self.TotalRideTime=self.TotalRideTime+self.pickuptime+self.droptime
            
            #Compute ride reward
            reward=self.reward_func(state,action,Time_matrix,1) #1 indicates ride completion
        else:
            #Update current location
            dropLocation=currLocation
            
            #Update totalHours and totaldays
            totalHours,totaldays=self.DayModifier(totalHours+1,totaldays)
            
            #Flag as 0 to indicate offline trip
            reward=self.reward_func(state,action,Time_matrix,0)  #Offline reward
            
        #Check if episode has ended 24*30 days=720, end of the month
        if self.TotalRideTime>=720:
            is_terminal=True
        next_state=(dropLocation,totalHours,totaldays) #state returned without encoding
        return next_state,action,reward,is_terminal

    #Update totalHours and totaldays
    def DayModifier(self, hour, nextday):
        """Time and week day modifier, Handling changing the week day based on time """
        while hour >= 24:
            if hour == 24:
                nextday = nextday+1
                hour = 0
            elif hour > 24:
                nextday = nextday+1
                hour = hour-24
            if nextday > 6:
                nextday = nextday-7
        return (hour, nextday)


    #Reset environment
    def reset(self):
        'Reseting the object and setting total commute as zero'
        self.TotalRideTime=0
        return self.action_space, self.state_space, self.state_init