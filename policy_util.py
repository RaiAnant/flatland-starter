from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool

from collections import defaultdict
import numpy as np
import os
import cv2
from datetime import datetime

# This action stops a train
STOP_ACTION = 4

def get_shortest_path(env, agent):
    '''
    Adopted from Anant.

    Returns the shortest path from a train position
    to the destination of that train.
    The path is a list of tuples: Tuple(train direction; train position).
    '''
    try:
        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
    except:
        possible_transitions = env.rail.get_transitions(*agent.initial_position, agent.direction)

    path = []

    min_distance = np.inf
    cur_direction = agent.direction
    cur_position = agent.position

    if cur_position is None:
        return []

    while min_distance != 0:

        min_distance = np.inf
        for direction in [(cur_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = get_new_position(cur_position, direction)
                if min_distance > env.distance_map.get()[agent.handle, new_position[0], new_position[1], direction]:
                    min_distance = env.distance_map.get()[agent.handle, new_position[0], new_position[1], direction]
                    next_pos = new_position
                    next_direction = direction

        cur_direction = next_direction
        cur_position = next_pos
        path.append((cur_direction, cur_position))
        possible_transitions = env.rail.get_transitions(*cur_position, cur_direction)

    return path


def get_best_action(agent, env, idx):
    '''
    Adopted from Anant.
    '''
    try:
        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
    except:
        possible_transitions = env.rail.get_transitions(*agent.initial_position, agent.direction)
    num_transitions = np.count_nonzero(possible_transitions)

    if num_transitions == 1:
        return 2
    else:
        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = get_new_position(agent.position, direction)
                min_distances.append(env.distance_map.get()[idx, new_position[0], new_position[1], direction])
            else:
                min_distances.append(np.inf)

        return np.argmin(min_distances) + 1

def path_blocked(agent, path, blocked_positions):
    '''
    Returns if a path that an agent wants to take is blocked or not.
    If any agent other than the given agent has registered the path in opposite direction, 
    returns True.
    
    agent : int
        The agent that checks if path is blocked.
    path : list(Tuple(direction, position))
        The path that is checked for obstacles
    blocked_positions : dict
        A dict with keys = Tuple(position, direction); values = set(agent_ids). Stores
        all positions/directions that have been registered by agents.
    '''
    for direction, pos in path:
        inverted_dir = (direction + 2) % 4
        key = (inverted_dir, pos)
        blocked_by = blocked_positions.get(key, {})
        if len(blocked_by) != 0 and agent not in blocked_by:
            return True
    return False


def set_blocked(agent, path, blocked_positions):
    '''
    Stores a path as blocked in blocked_positions.
    '''
    for direction, pos in path:
        if pos is None: 
            continue
        blocked_positions[(direction, pos)] = blocked_positions.get(pos, set()) | {agent}
    return blocked_positions


def unblock_position(agent, pos, direction, blocked_positions):
    '''
    Deletes a registration for pos/direction of the given agent.
    This is called ech time an agents moves to remove the registration
    from the previous position.
    '''
    key = (direction, pos)
    if key in blocked_positions:
        blocked_positions[key].discard(agent)
        #if len(blocked_positions[key]) == 0:
        #    del blocked_positions[key]
    return blocked_positions


def my_controller(obs, env, blocked_positions, stop_agents=True, return_full_path=False):
    '''
    Controller that navigates agents through the environment.
    Here is how it works:
    1 - Each agent calculates the shortest path to the destination
    2 - If (path_blocked == True) the path has a conflicting registration from a different agent 
            * Stop this agent and don't register its path
      - If (path_blocked == False) the path has no conflicting registration from a different agent
            * Register that path and choose the best action.
            * Unblock the current position, because it will be clear after the action has been performed
    3 - Repeat until the maximal time of environment is reached, or all agents have arrived at 
        their respective destinations.
    '''
    actions = {}
    paths = {}

    for idx, agent in enumerate(env.agents):
        path = get_shortest_path(env, agent)  # we need to save paths to avoid this
        paths[idx] = path
        #if idx == 0:
        #    print(path)
        if stop_agents and path_blocked(agent.handle, path, blocked_positions):
            actions[idx] = STOP_ACTION  # stop the agent
        else:
            actions[idx] = get_best_action(agent, env, idx)
            blocked_positions = set_blocked(agent.handle, path, blocked_positions)  # we probably only want to do this one time so this is not the best way
            #print(blocked_positions)
            blocked_positions = unblock_position(agent.handle, agent.position, agent.direction, blocked_positions)

    if return_full_path:
        return actions, paths

    return actions


class EnvironmentAdapter:
    '''
    This environment wraps the main principles of the path registration/blocked path approach:
    Paths of agents are automatically registered and the observations this environment provides 
    directly correspond to the registered paths of the agents.

    Observations:
    1 - The registered positions are encoded via _encode_positions
    2 - The path for each agent is encoded:
            Each position in path is represented as two lists of length 4; Example:
            Self  [0, 1, 0, 0]
            Other [1, 0, 0, 1]
            Meaning that the agent it self traverses this position in direction 1
            Other agents also traverse this position in directions 0 and 4.
            This observation should be considered a blockage, because the agent encounters other agents 
            in opposite direction.
            This encoding is done in method create_agent_observation

    Step method (env_step):
    1 - Every agent registeres a path (s. agents_register_tiles) 
        1.1 - If first_obs, the registration will be done only once after the respective agent spawned.
        1.2 - The agent unregisteres the current position if the action is not STOP_ACTION
    2 - Pass the actions to the environment and simulate.
    '''

    def __init__(self, first_obs, max_agents):
        self.remote_client = DummyRemoteClient(max_agents)
        self.blocked_positions = {}
        self.renderer = None
        self.first_obs = first_obs

    @property
    def env(self):
        try:
            return self.remote_client.env
        except:
            print('Remote client has not been initialized. '
                  'Call env_create to initialize it.')

    @property
    def max_agents(self):
        return self.remote_client.max_agents

    def get_shortest_paths(self, block_on_registered=True):
        actions = {}

        for agent in self.env.agents:
            agent_idx = agent.handle

            if block_on_registered:
                path = get_shortest_path(self.env, agent)

                if self._path_blocked(agent.handle, path):
                    actions[agent_idx] = STOP_ACTION  # stop the agent
            else:
                actions[agent_idx] = get_best_action(agent, self.env, agent_idx)

        return actions

    def _encode_positions(self):
        '''
        The registered positions are encoded as 1-hot vectors of length 4, 
        each index encodes one orientation of an agent (4 possible orientations):
        [North, East, South, West]
        For a north-bound agent, the encoding would be [1, 0, 0, 0].
        '''
        n_agents = self.remote_client.max_agents
        encodings = defaultdict(lambda: np.zeros([n_agents, 4]))
        dir2hot = np.eye(4)

        # for each currently blocked cell
        for direction, pos in self.blocked_positions:
            blocking_agents = list(self.blocked_positions[(direction, pos)])
            # for each agent that is blocking this cell
            for block in blocking_agents:
                encodings[pos][block] = dir2hot[direction]

        return encodings

    def _register_along_path(self, agent, path):
        '''
        Stores each position/direction of agent along the given path
        in self.blocked_positions.
        '''
        path_without_station = path[:-1]
        for direction, pos in path_without_station:
            if pos is None: 
                continue
            key = (direction, pos)
            registered_pos = self.blocked_positions.get(key, set()) | {agent}
            self.blocked_positions[key] = registered_pos
        return self.blocked_positions

    def _unregister_along_path(self, agent, path):
        '''
        Deletes each position/direction of agent along the given path
        from self.blocked_positions.
        '''
        path_without_station = path[:-1]
        for direction, pos in path_without_station:
            if pos is None: 
                continue
            key = (direction, pos)
            self._unregister_position(agent, pos, direction)
        return self.blocked_positions

    def _unregister_position(self, agent, agent_pos, agent_dir):
        '''
        Deletes the given position/direction of agent
        from self.blocked_positions.
        '''
        key = (agent_dir, agent_pos)
        if key in self.blocked_positions:
            self.blocked_positions[key].discard(agent)
            if len(self.blocked_positions[key]) == 0:
                del self.blocked_positions[key]
        return self.blocked_positions

    def _path_blocked(self, agent, path):
        '''
        Returns true if the given path is blocked for the given agent.
        '''
        for direction, pos in path:
            inverted_dir = (direction + 2) % 4
            key = (inverted_dir, pos)
            blocked_by = self.blocked_positions.get(key, {})
            if len(blocked_by) != 0 and agent not in blocked_by:
                return True
        return False

    def env_create(self, obs_builder_object=None):
        '''
        Creates a new environment and performs reset.
        '''
        obs, info = self.remote_client.env_create(obs_builder_object=obs_builder_object)
        self.renderer = None
        return self.reset()

    def reset(self, random_seed=None):
        '''
        Removes all agent registrations and resets the current environment.
        '''
        self.blocked_positions = {}
        self.agents_registered = defaultdict(lambda: False)
        obs, info = self.env.reset(random_seed=random_seed)
        agent_obs, paths = self.agents_register_tiles()
        return agent_obs, info

    def render(self):
        '''
        Renders the environment and saves the image in images_out/images/.
        TODO: Include parameter to disable the saving of images.
        '''
        if self.renderer is None:
            self.renderer = RenderTool(self.env)

        img = self.renderer.render_env(show=False,
                                      show_inactive_agents=False,
                                      show_predictions=True,
                                      show_observations=True,
                                      frames=True,
                                      show_rowcols=True,
                                      return_image=True)
        #img = cv2.resize(img, (img.shape[0] * 2, img.shape[1]*2), interpolation=cv2.INTER_NEAREST)
        tile_width = img.shape[1] // self.env.width
        tile_height = img.shape[0] // self.env.height

        time_str = datetime.now().isoformat()
        timestamp = datetime.now().timestamp()

        out_dir = f'images_out/images/{time_str}/'

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        cv2.imwrite(os.path.join(out_dir, f'{timestamp}.jpg'), img)

        cv2.imshow('Flatland', img)
        cv2.waitKey(0)

    def create_agent_observation(self, path, agent):
        cell_encodings = self._encode_positions()
        n_agents = self.remote_client.max_agents
        all_agents = list(range(n_agents))

        if len(path) == 0:
            # if an agent didn't yet depart or has reached its target
            # the agent observation is None
            return None

        other_agents = all_agents[:agent] + all_agents[agent+1:]
        path_features = np.array(
            [ [ cell_encodings[p][agent], 
                cell_encodings[p][other_agents].max(0) ] 
                for d, p in path ])
        return path_features

    def agents_register_tiles(self, actions=None):
        '''
        For every agent, registeres all positions along the shortest path
        to the respective destination.
        '''
        paths = {}
        agent_obs = {}
        prior_registrations = self.agents_registered.copy()

        # ToDo: check idx vs agent.handle
        for agent in self.env.agents:
            agent_id = agent.handle
            agent_pos = agent.position
            agent_dir = agent.direction

            if self.first_obs and self.agents_registered[agent_id]:
                continue

            path = get_shortest_path(self.env, agent)
            paths[agent_id] = path
            #actions[idx] = get_best_action(agent, self.env, idx)

            if not self.agents_registered[agent_id] and len(path) > 0 \
                :#and not self._path_blocked(agent_id, path):
                self._register_along_path(agent_id, path)
                self.agents_registered[agent_id] = True

            #if actions is not None and actions[agent_id] == STOP_ACTION:
            #    print('Unregister path agent', agent_id)
            #    self._unregister_along_path(agent_id, path)

            if actions is None or actions[agent_id] != STOP_ACTION:
                self._unregister_position(agent_id, agent_pos, agent_dir)

        for idx, path in paths.items():
            if self.first_obs and prior_registrations[idx]:
                agent_obs[idx] = None
                continue
            agent_obs[idx] = self.create_agent_observation(path, idx)

        return agent_obs, paths#, actions, paths

    def env_step(self, actions):
        '''
        Performs a step in the environment.
        '''
        agent_obs, paths = self.agents_register_tiles(actions)
        obs, all_rewards, done, info = self.remote_client.env_step(actions)

        # update info dict
        info['raw_observation'] = obs
        info['all_rewards'] = all_rewards
        info['done'] = done

        return agent_obs, all_rewards, done, info


class DummyRemoteClient:
    '''
    Environment wrapper to simulate the remote client used in model evaluation 
    that runs on the servers of the Flatland challenge hosts.
    This circumvents having to run an own server locally.
    '''

    # Initialize the properties of the environment
    speed_dist = {1: 0.4, 0.5: 0.3, 0.3: 0.3}

    def __init__(self, max_agents):
        self.max_agents = max_agents
        self.n_agents = np.random.randint(2, self.max_agents)
        self.width = np.random.randint(26, 38)
        self.height = int(self.width * np.clip(np.random.normal(1, 1), 1.0, 1.3))
        self.n_cities = np.random.randint(1, 3)

    def env_step(self, actions):
        return self.env.step(actions)

    def env_create(self, obs_builder_object):
        SEED = None
        self.env = RailEnv(
            width=self.width,
            height=self.height,
            number_of_agents=self.n_agents,
            rail_generator=sparse_rail_generator(#max_num_cities=self.n_cities,
                                                 grid_mode=True,
                                                 max_rails_between_cities=1,
                                                 max_rail_pairs_in_city=2,
                                                 seed=SEED),
            line_generator=sparse_line_generator(),
            obs_builder_object=obs_builder_object, random_seed=SEED
        )
        return self.env.reset()




if __name__ == '__main__':
    from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv

    my_observation_builder = GlobalObsForRailEnv()
    remote_client = EnvironmentAdapter()
    obs, info = remote_client.env_create(obs_builder_object=my_observation_builder)

    episode = 0

    while True:

        print("==============")
        episode += 1
        print("[INFO] EPISODE_START : {}".format(episode))
        # NO WAY TO CHECK service/self.evaluation_done in client

        obs, info = remote_client.env_create(obs_builder_object=my_observation_builder)
        env_renderer = RenderTool(remote_client.env)

        if not obs:
            """
            The remote env returns False as the first obs
            when it is done evaluating all the individual episodes
            """
            print("[INFO] DONE ALL, BREAKING")
            break

        step = 0
        while True:
            #action = my_controller(obs, remote_client.env, blocked_positions)
            actions = remote_client.get_shortest_paths()
            try:
                observation, all_rewards, done, info = remote_client.env_step(actions)
            except:
                print("[ERR] DONE BUT step() CALLED")

            if (True):  # debug
                print("-----")
                # print(done)
                print("[DEBUG] REW: ", all_rewards)
            
            
            if True:
                img_r = env_renderer.render_env(show=False,
                                              show_inactive_agents=False,
                                              show_predictions=True,
                                              show_observations=True,
                                              frames=True,
                                              show_rowcols=True,
                                              return_image=True)
                #img_r = cv2.resize(img, (img.shape[0] * 2, img.shape[1]*2), interpolation=cv2.INTER_NEAREST)
                tile_width = img_r.shape[1] // remote_client.env.width
                tile_height = img_r.shape[0] // remote_client.env.height

                if not os.path.isdir('images_out/images/'):
                    os.makedirs('images_out/images/')

                cv2.imwrite("images_out/images/" + str(step).zfill(4) + ".jpg", img_r)

                cv2.imshow('Flatland', img_r)
                cv2.waitKey(0)
            
            step += 1

            # break
            if done['__all__']:
                print("[INFO] EPISODE_DONE : ", episode)
                print("[INFO] TOTAL_REW: ", sum(list(all_rewards.values())))
                break


    print("Evaluation Complete...")
    print(remote_client.submit())
