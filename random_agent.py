from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.core.grid.grid4_utils import get_new_position
import numpy as np

remote_client = FlatlandRemoteClient()


def get_shortest_path(env, agent):
    try:
        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
    except:
        possible_transitions = env.rail.get_transitions(*agent.initial_position, agent.direction)

    path = []

    min_distance = np.inf
    cur_direction = agent.direction
    cur_position = agent.position

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


def get_best_action(env, idx, agent):
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


def path_blocked(path, blocked_positions):  # TODO: complete this function
    return False


def set_blocked(path, blocked_positions):  # TODO: complete this function
    pass


def unblock_position(pos, blocked_positions):  # TODO: complete this function
    pass


def my_controller(obs, env, blocked_positions):
    actions = {}
    for idx, agent in enumerate(env.agents):
        path = get_shortest_path(env, agent)  # we need to save paths to avoid this
        if path_blocked(path, blocked_positions):
            actions[idx] = 4  # stop the agent
        else:
            actions[idx] = get_best_action(env, idx, agent)
            set_blocked(path, blocked_positions)  # we probably only want to do this one time so this is not the best way
            unblock_position(agent.position, blocked_positions)

    return actions


# my_observation_builder = TreeObsForRailEnv(
#     max_depth=3, predictor=ShortestPathPredictorForRailEnv())
my_observation_builder = GlobalObsForRailEnv()

episode = 0
blocked_positions = {}
while True:

    print("==============")
    episode += 1
    print("[INFO] EPISODE_START : {}".format(episode))
    # NO WAY TO CHECK service/self.evaluation_done in client

    obs, info = remote_client.env_create(obs_builder_object=my_observation_builder)
    if not obs:
        """
        The remote env returns False as the first obs
        when it is done evaluating all the individual episodes
        """
        print("[INFO] DONE ALL, BREAKING")
        break

    while True:
        action = my_controller(obs, remote_client.env, blocked_positions)
        try:
            observation, all_rewards, done, info = remote_client.env_step(
                action)
        except:
            print("[ERR] DONE BUT step() CALLED")

        if (True):  # debug
            print("-----")
            # print(done)
            print("[DEBUG] REW: ", all_rewards)
        # break
        if done['__all__']:
            print("[INFO] EPISODE_DONE : ", episode)
            print("[INFO] TOTAL_REW: ", sum(list(all_rewards.values())))
            break

print("Evaluation Complete...")
print(remote_client.submit())
