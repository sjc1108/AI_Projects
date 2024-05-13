import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env = gym.envs.make("FrozenLake-v1")
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):

            if random.uniform(0, 1) < EPSILON: #this is eps_greedy
                action = env.action_space.sample() # currently only performs a random action.
            else: #use qtable for pred
                pred = np.array([Q_table[(obs, action_i)] for action_i in range(env.action_space.n)])
                action = np.argmax(pred) #hightest q val

            new_obs,reward,terminated,truncated,info = env.step(action) #steeping

            done = terminated or truncated #if epi is over or not

            #update w/q learning eqn
            if not done:
                best_future_q = max([Q_table[(new_obs, action_i)] for action_i in range(env.action_space.n)])
                Q_table[(obs, action)] = (1- LEARNING_RATE) * Q_table[(obs, action)] + LEARNING_RATE * (reward + DISCOUNT_FACTOR* best_future_q)
            else:
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs,action)] + LEARNING_RATE * reward

            obs = new_obs #update w/ curr
            episode_reward += reward # update episode reward


        #for decay ep
        EPSILON *= EPSILON_DECAY
           # done = terminated or truncated
         
         
            
            
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################