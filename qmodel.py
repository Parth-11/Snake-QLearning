import numpy as np
from snake_game import Env, FOOD_REWARD, SIZE_X, SIZE_Y,SELF_PENALTY,BOUNDARY_PENALTY
import matplotlib.pyplot as plt
from matplotlib import style
import time
import pickle

style.use('ggplot')

LEARNING_RATE = 0.1
DISCOUNT = 0.95

epsilon = 1
EPS_DECAY = 0.9998

EPISODES = 250000
SHOW_EVERY  = 2500
MOVES_PER_EP = 200

start_q_table = None

if start_q_table is None:
    q_table = {}

    for x1 in range(0,SIZE_X):
        for y1 in range(0,SIZE_Y):
            for x2 in range(0,SIZE_X):
                for y2 in range(0,SIZE_Y):
                    for d1 in range(0,2):
                        for d2 in range(0,2):
                            for d3 in range(0,2):
                                for d4 in range(0,2):
                                    q_table[((x1,y1),((x2,y2)),(d1,d2,d3,d4))] = [np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q_table,'rb') as f:
        q_table = pickle.load(f)

# print(q_table)

epsiode_rewards = []
env = Env()

for episode in range(EPISODES):

    if episode%SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(epsiode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    
    epsiode_reward = 0

    state = env.reset()

    #print(f"Initial State: {state}")
    
    for i in range(MOVES_PER_EP):

        if np.random.rand() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0,4)

        # action = np.argmax(q_table[state])

        new_state,reward,done = env.step(action)

        #print(f"State {i}: {new_state}")

        if done:
            break

        max_future_q = np.max(q_table[new_state])
        curr_q = q_table[state][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == BOUNDARY_PENALTY:
            new_q = -BOUNDARY_PENALTY
        elif reward == SELF_PENALTY:
            new_q = -SELF_PENALTY
        else:
            new_q = (1-LEARNING_RATE)*curr_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[state][action] = new_q

        if show:
            env.render()
            time.sleep(0.05)
        
        epsiode_reward+= reward
    
    epsiode_rewards.append(epsiode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(epsiode_rewards,np.ones((SHOW_EVERY,))//SHOW_EVERY,mode='valid')

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}")
plt.xlabel("Episode #")
plt.show()

with open(f"q-table{int(time.time())}.pickle",'wb') as f:
    pickle.dump(q_table,f)

env.close()
