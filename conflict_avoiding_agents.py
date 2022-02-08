from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.utils.rendertools import RenderTool
from policy_util import my_controller, DummyRemoteClient
import cv2

import numpy as np

#remote_client = FlatlandRemoteClient()
remote_client = DummyRemoteClient(10)


# my_observation_builder = TreeObsForRailEnv(
#     max_depth=3, predictor=ShortestPathPredictorForRailEnv())
my_observation_builder = GlobalObsForRailEnv()
episode = 0

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

    blocked_positions = {}

    env_renderer = RenderTool(remote_client.env)

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

        
        if True:
            img_r = env_renderer.render_env(show=False,
                                          show_inactive_agents=False,
                                          show_predictions=True,
                                          show_observations=True,
                                          frames=True,
                                          show_rowcols=True,
                                          return_image=True)
            mark_agents = None#stop_agents if stop_agents is not None else delay_agents
            #img_r = cv2.resize(img, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)
            tile_width = img_r.shape[1] // remote_client.env.width
            tile_height = img_r.shape[0] // remote_client.env.height
            #cv2.putText(img_r, 'Time: %d' % step, (20, 50), fontFace=0, fontScale=1, color=1, thickness=3)
            if mark_agents is not None:
                for sa in range(len(mark_agents['agent'])):
                    #print(sa, stopping_countdown, mark_agents['agent'][sa])
                    current_pos = remote_client.env.agents[mark_agents['agent'][sa]].position
                    if current_pos is not None:
                    #cv2.putText(img_r, 'Pos: (%d, %d)  Time: %d' % (*env.agents[0].position, step), (20, 50), fontFace=0, fontScale=1, color=1, thickness=3)
                        img_r = cv2.circle(img_r, (int((current_pos[1] + 0.5) * tile_width), 
                                                   int((current_pos[0] + 0.5) * tile_height)),
                                           radius=tile_width//2, color=1, thickness=3)
            #else:
            #    cv2.putText(img_r, 'Pos: (-, -)', (20, 50), fontFace=0, fontScale=1, color=1, thickness=3)

            cv2.imshow('Flatland', img_r)
            cv2.waitKey(0)
        
        # break
        if done['__all__']:
            print("[INFO] EPISODE_DONE : ", episode)
            print("[INFO] TOTAL_REW: ", sum(list(all_rewards.values())))
            break


print("Evaluation Complete...")
print(remote_client.submit())
