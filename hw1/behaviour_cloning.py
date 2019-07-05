import tensorflow as tf
import pickle
import mujoco_py
import gym
import numpy as np

def import_expert_data(expert):
    with open('./expert_data/{}.pkl'.format(expert), 'rb') as f:
        Data = pickle.load(f)
        return Data


def process_expert_data(expert):
    Data= import_expert_data(expert)
    observations= Data["observations"]
    actions = Data["actions"]
    actionShape= actions.shape
    if len(actionShape)> 2:
        # reshape actions to be 2d
        try:
          actions = actions.reshape([actionShape[0], actionShape[-1]])
        except:
            print("action cannot be reshaped")
    return observations,actions

def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()

def make_model(observations,actions):
    
    #creating input and output placeholders
    input_ph=tf.placeholder(dtype=tf.float32, shape=(None, observations.shape[1]))
    output_ph=tf.placeholder(dtype=tf.float32, shape=(None,actions.shape[1]))

    # creating variable
    W0=tf.get_variable(name='W0', shape=[376,500], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    W1=tf.get_variable(name='W1', shape=[500,1000], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    W2=tf.get_variable(name='W2', shape=[1000,17], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    B0=tf.get_variable(name='B0', shape=[500], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    B1=tf.get_variable(name='B1', shape=[1000], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    B2 = tf.get_variable(name='B2', shape=[17], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    weights=[W0, W1, W2]
    bias=[B0, B1, B2]
    activation=[tf.nn.relu, tf.nn.relu, None]

    #creating the graph
    layer = input_ph
    for W,B,A in zip(weights,bias,activation):
        layer = tf.matmul(layer,W)+B
        if A is not None:
            layer=A(layer)
    output_pred=layer
    
    return input_ph, output_ph, output_pred


# Training a created model
def train_model( observations, actions):

    sess=tf_reset()
    input_ph, output_ph, output_pred = make_model(observations,actions)

    # creating the loss function
    loss= tf.reduce_mean(tf.square(output_ph-output_pred))

    # creating an optimizer
    opt=tf.train.AdamOptimizer().minimize(loss)

    # initiate the varaibles
    sess.run(tf.global_variables_initializer())

    #create a saver for the model
    saver= tf.train.Saver()

    #run_training
    batch_size= 128
    for step in  range (10000):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=len(observations), size=batch_size)
        input_batch = observations[indices]
        output_batch = actions[indices]

        # run the optimizer and get the mse
        _, mse_run = sess.run([opt, loss], feed_dict={input_ph: input_batch, output_ph: output_batch})

        # print the mse every so often
        if step % 1000 == 0:
            print('{0:04d} mse: {1:.3f}'.format(step, mse_run))
            saver.save(sess, "/home/aditya/external courses/berkely:deep reinforcment learning/homework/hw1/model/bc_model.ckpt")



def eval_model(observations, actions, expert, rollouts):
    sess = tf_reset()

    #make the graph of the model so it can be restorerd
    input_ph, output_ph, output_pred = make_model(observations,actions)

    saver = tf.train.Saver()
    saver.restore(sess, "/home/aditya/external courses/berkely:deep reinforcment learning/homework/hw1/model/bc_model.ckpt")

    #make the gym environment
    env = gym.make(expert)

    #initialize the arrays to store obs actions and rewards for each rollout
    eval_observations=[]
    eval_actions=[]
    returns=[]

    for i in range (rollouts):
        # reset the env and obtain the first observation
        obs = env.reset()
        step = 0
        done = False
        total_reward = 0
        while not done or step == 5000:
            obs = obs.reshape([1,obs.shape[0]])
            pred_action = sess.run(output_pred, feed_dict={input_ph: obs})
            eval_actions.append(pred_action)
            eval_observations.append(obs)
            obs, reward, done, info = env.step(pred_action)

def main ():
    observations,actions = process_expert_data("Humanoid-v2")
    train_model(observations, actions)
    eval_observations,eval_actions,returns,avg_return=eval_model(observations,actions,"Humanoid-v2",20)
    print("returns=", returns, "avg return=", avg_return)

if __name__ == '__main__':
    main()












