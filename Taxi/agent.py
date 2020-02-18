import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=0, gamma=0.99, alpha=0.2):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        p = np.full(self.nA, (1/self.nA))
        if state in self.Q.keys():
            p = np.full(self.nA, (self.epsilon/(self.nA-1)))
            p[np.argmax(self.Q[state])] = (1-self.epsilon)
        return np.random.choice(self.nA, p=p)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1
        new_Qsa = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] =((1 - self.alpha)*self.Q[state][action])+( self.alpha*(new_Qsa - self.Q[state][action])  )