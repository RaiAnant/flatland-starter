from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.step_utils.states import TrainState
from policy_util import EnvironmentAdapter, my_controller, STOP_ACTION

import numpy as np
from collections import defaultdict
import argparse
import os

import tensorflow as tf


# enable GPU growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


SEED = 1234

# ----------------- Setup -----------------
# THRESHOLD to convert probabilities to action (probs above this threshold leads to action GO)
THRESHOLD = 0.5
# Max number of timesteps (used when padding observations to uniform lengths.
MAX_TIME = 60
# This depends on the size of the observations of the environment (maybe move it as a property of environment?)
N_FEATURES = 2 * 4
# -----------------------------------------


def model_predict(model, observation):
    '''
    Convert model probability to prediction.
    '''
    probability = model(observation[tf.newaxis])
    print('Agent GO probability:', probability.numpy()[0][0])
    return probability[0][0] > THRESHOLD

def run_episode(first_obs=False, model=None, render=False, max_agents=5):
    '''
    Run an episode in the environment.
    if first_obs is True, return only the first observation of each agent.
    if model is given, use the actions that the model predicts.
    '''
    my_observation_builder = GlobalObsForRailEnv()
    remote_client = EnvironmentAdapter(first_obs=first_obs, max_agents=max_agents)

    agent_obs, info = remote_client.env_create(obs_builder_object=my_observation_builder)
    n_agents = remote_client.max_agents
    run_info = { 'max_agents' : n_agents }

    agent_observations = []
    outcomes = []
    model_prediction = {}

    while True:
        action = remote_client.get_shortest_paths(block_on_registered=False)

        if model is not None:
            for a, a_obs in agent_obs.items():

                if a_obs is None:
                    continue

                a_obs = preprocess_observation(a_obs)

                if not model_predict(model, a_obs):
                    action[a] = STOP_ACTION
                    model_prediction[a] = True

        agent_obs, all_rewards, done, info = remote_client.env_step(action)
        agent_observations.append(agent_obs)
        outcomes.append((all_rewards, done))
        
        if render:
            remote_client.render()

        if done['__all__']:
            agent_done = [ a.state == TrainState.DONE 
                           for a in remote_client.env.agents ]
            return agent_observations, agent_done, run_info


def get_agent_initial_observation(raw_observations, agent):
    a_obs = next(filter(lambda o: agent in o and o[agent] is not None, raw_observations))[agent]
    a_obs = preprocess_observation(a_obs)
    return a_obs

def preprocess_observation(a_obs):
    '''
    Reshaped the given observation to vector form and
    pads the observation to a uniform length time series.
    '''
    a_obs = np.reshape(a_obs, [len(a_obs), -1])
    n_timesteps = len(a_obs)
    return np.pad(a_obs, 
        [(MAX_TIME-n_timesteps, 0), (0, 0)], 
        'constant')

def cached_generator(n_samples, max_agents, data_path=None):
    '''
    Creates a data generator with a cache: Once the dataset has been iterated once, the
    generator falls back to the cache.
    '''
    samples = []

    if data_path is not None and os.path.exists(data_path):
        samples = np.load(data_path, allow_pickle=True)

    def data_generator():
        i = 0

        if len(samples) != 0:
            np.random.shuffle(samples)

            for x in samples:
                yield tuple(x)
            return

        while i < n_samples:
            obs, agent_done, info = run_episode(first_obs=True, max_agents=max_agents)
            n_agents = len(agent_done)
            n_timesteps = len(obs)

            # yield and observation / agent-done pair for each agent
            for a in range(n_agents):
                try:
                    a_obs = get_agent_initial_observation(obs, a)
                except StopIteration:
                    # agent never spawned
                    continue
                i += 1

                yield a_obs, [int(agent_done[a])], n_agents, n_timesteps
                samples.append((a_obs, [int(agent_done[a])], n_agents, n_timesteps))
        
        if data_path is not None:
            np.save(data_path, samples)

    return data_generator

def build_model(n_timesteps, n_features, ckpt=None):
    observation_input = tf.keras.layers.Input(shape=(n_timesteps, n_features))

    # Either handle temporal data via GRU
    x = tf.keras.layers.GRU(16, use_bias=False)(observation_input)
    # Or handle it by squashing the temporal axis and apply MLP
    #x = tf.keras.layers.GlobalMaxPooling1D()(observation_input)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model =  tf.keras.Model(inputs=[observation_input], outputs=[output])

    if ckpt is not None:
        checkpoint = tf.train.Checkpoint(net=model)
        checkpoint.restore(tf.train.latest_checkpoint(ckpt))
        print('Restored checkpoint')

    return model

def build_dataset(batch_size, batches_per_epoch, max_agents, data_path=None):
    data_gen = cached_generator(batch_size * batches_per_epoch, max_agents, data_path=data_path)
    dataset = tf.data.Dataset.from_generator(data_gen, 
        output_signature=(
            tf.TensorSpec([None, 8], dtype=tf.float32),
            tf.TensorSpec([None], dtype=tf.int64),
            tf.TensorSpec(None, dtype=tf.int64),
            tf.TensorSpec(None, dtype=tf.int64)),
        ) 
    dataset = dataset.map(lambda obs, agent_done, n_agents, n_timesteps: (obs, agent_done))
    return dataset.batch(batch_size, drop_remainder=True).prefetch(2)

def train_discriminator(dataset, epochs, ckpt=None):
    model = build_model(MAX_TIME, N_FEATURES, ckpt=ckpt)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer, 
        loss=tf.keras.losses.binary_crossentropy, 
        metrics=['acc', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    model.fit(dataset, epochs=epochs)
    checkpoint = tf.train.Checkpoint(net=model, optimizer=optimizer)

    if ckpt is None:
        ckpt = 'ckpts/model_5_epochs'
    if not os.path.isdir(os.path.dirname(ckpt)):
        os.makedirs(os.path.dirname(ckpt))

    ckpt_manager = tf.train.CheckpointManager(checkpoint, ckpt, max_to_keep=3)
    print(ckpt_manager.save())

def run_discriminator(max_agents, ckpt=None):
    model = build_model(MAX_TIME, N_FEATURES, ckpt=ckpt)
    run_episode(first_obs=False, model=model, render=True, max_agents=max_agents)

# predict outcome if the agent goes 
# if 0 --> stop
# if 1 --> go

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
        description='Train and run models on the Flatland environment.')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--run',   action='store_true', default=False)
    parser.add_argument('--epochs', '-e', type=int, nargs='?', default=15,
                        help='The number of epochs to train, defaults to 50.')
    parser.add_argument('--checkpoint', '-c', type=str, nargs='?', 
                        help='Path to a checkpoint to continue training '
                             'or run the model from there.')
    parser.add_argument('--data_path', '-d', type=str, nargs='?', 
                        help='Path to load or save the training data.')

    # ----------- Setup an experiment with these args -----------
    parser.add_argument('--batches_per_epoch', '-s', type=int, nargs='?', default=1000,
                        help='Number of batches of an epoch.')
    parser.add_argument('--batch_size', '-b', type=int, nargs='?', default=10,
                        help='The size of every batch.')
    parser.add_argument('--max_agents', '-m', type=int, nargs='?', default=20,
                        help='The maximal number of agents in the environment.')

    args = parser.parse_args()

    config = vars(args)

    if config['run']:
        run_discriminator(
            config['max_agents'], 
            ckpt=config['checkpoint'])
    else:
        dataset = build_dataset(
            config['batch_size'], 
            config['batches_per_epoch'], 
            config['max_agents'], 
            data_path=config['data_path'])
        train_discriminator(dataset, config['epochs'], ckpt=config['checkpoint'])