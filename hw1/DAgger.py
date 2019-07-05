from behaviour_cloning import *
import tensorflow as tf
import pickle
import mujoco_py
import gym
import numpy as np

def main():
    observations, actions = process_expert_data("Humanoid-v2")
    comp_returns= np.zeros(shape=(20,20))
    comp_avg_return=[]
    for i in range (20):
        train_model(observations, actions)
        eval_observations, eval_actions, returns, avg_return = eval_model(observations, actions, "Humanoid-v2", 20)
        print("returns=", returns, "avg return=", avg_return)
        observations=np.concatenate((observations,eval_observations), axis=0)
        actions=np.concatenate((actions,eval_actions), axis=0)
        comp_returns[i]=returns
        comp_avg_return.append(avg_return)
    print("complete avg returns",comp_avg_return)

if __name__ == '__main__':
    main()
