import torch as th
import tianshou as ts
import Train
import numpy as np
import os
import MainUnits
import gym
from Enviroments.envs import EnvUnits


def evaluation(policyfile, partfile, custom_env, action_file):
    SP = MainUnits.SpherePoints()
    spherepoints = SP.discretize_sphere()
    graphs, max_bfs_normals, base_face_ids = EnvUnits.ReadDataFromPath_v2(partfile,Mode="PNAConv",EnvMode="without_BF")
    env = gym.make(custom_env,path_of_part=partfile, sphere=spherepoints, graphs=graphs, max_bfs_normals = max_bfs_normals,  base_face_ids = base_face_ids)

    face_tags = []
    files = os.listdir(partfile)
    temp_file = partfile + '\\' + files[0]
    with open(temp_file) as p:
        lines = p.readlines()
        for i in range(len(lines)):
            data = list(map(str, (lines[i].strip().split(' '))))
            if data[0] == "#N":
                face_tags.append(data[-1])

    policy = Train.construct_policy_PPO(env)
    policy.load_state_dict(th.load(policyfile))
    policy_network = policy.actor
    policy_network.eval()

    face_tags = []
    files = os.listdir(partfile)
    temp_file = partfile + '\\' + files[0]
    with open(temp_file) as p:
        lines = p.readlines()
        for i in range(len(lines)):
            data = list(map(str, (lines[i].strip().split(' '))))
            if data[0] == "#N":
                face_tags.append(data[-1])  # Tag

    env.assignment_id(0)
    temp_graph = env.reset()

    if th.sum(temp_graph.dstdata["NodeAttr"][:, -2]) >= 1:
        base_ids = th.where(temp_graph.dstdata["NodeAttr"][:, -2] == 1)[0]
        wether_base_face = 1
    else:
        wether_base_face = 0

    if wether_base_face == 1:
        with open(action_file, 'w') as f:
            for base_id_index in range(base_ids.shape[0]):
                all_obs = []
                all_actions = []
                base_id = base_ids[base_id_index]
                temp_graph = env.reset()
                temp_obs = np.array([temp_graph])
                k = 0
                while (True):
                    if k == 0:
                        action = th.tensor([base_id])[0]
                        temp_obs, reward, done, _ = env.step(action)
                        temp_obs = np.array([temp_obs])
                        k = 1
                    else:
                        reward = 0
                        num = 0
                        lgts = policy_network(temp_obs)[0].squeeze(0)
                        lgts_copy = lgts.clone()
                        lgts_sort = th.sort(lgts_copy, descending=True)

                        while (reward == 0):
                            action_lgts = lgts_sort[0][num]
                            if action_lgts == 0:
                                break
                            action = th.where(lgts == action_lgts)[0][0]
                            temp_obs, reward, done, _ = env.step(action)
                            temp_obs = np.array([temp_obs])
                            num += 1

                    if reward == 0:
                        break
                    else:
                        all_obs.append(temp_obs)
                        all_actions.append(int(action))

                for a in all_actions:
                    f.write(str(face_tags[a]))
                    f.write('\t')
                f.write('\n')
    else:
        with open(action_file, 'w') as f:
            temp_obs = np.array([temp_graph])
            faces_mask = np.zeros(temp_graph.num_nodes())

            num_of_feature = 0
            while (np.sum(faces_mask) != temp_graph.num_nodes()):
                all_obs = []
                all_actions = []
                if num_of_feature == 0:
                    num_of_feature = 1
                    while (True):
                        reward = 0
                        num = 0
                        lgts = policy_network(temp_obs)[0].squeeze(0)
                        lgts_copy = lgts.clone()
                        lgts_sort = th.sort(lgts_copy, descending=True)
                        while (reward == 0):
                            action_lgts = lgts_sort[0][num]
                            if action_lgts == 0:
                                break
                            action = th.where(lgts == action_lgts)[0][0]
                            temp_obs, reward, done, _ = env.step(action)
                            temp_obs = np.array([temp_obs])
                            num += 1

                        if reward == 0:
                            break
                        else:
                            all_obs.append(temp_obs)
                            all_actions.append(int(action))
                            faces_mask[int(action)] = 1
                    for a in all_actions:
                        f.write(str(face_tags[a]))
                        f.write('\t')
                    f.write('\n')
                else:
                    last_faces = np.where(faces_mask == 0)[0]
                    for face_id_index in range(last_faces.shape[0]):
                        all_obs = []
                        all_actions = []
                        face_id = last_faces[face_id_index]
                        temp_graph = env.reset()
                        temp_obs = np.array([temp_graph])
                        k = 0
                        while (True):
                            if k == 0:
                                action = th.tensor([face_id])[0]
                                temp_obs, reward, done, _ = env.step(action)
                                temp_obs = np.array([temp_obs])
                                k = 1
                            else:
                                reward = 0
                                num = 0
                                lgts = policy_network(temp_obs)[0].squeeze(0)
                                lgts_copy = lgts.clone()
                                lgts_sort = th.sort(lgts_copy, descending=True)

                                while (reward == 0):
                                    action_lgts = lgts_sort[0][num]
                                    if action_lgts == 0:
                                        break
                                    action = th.where(lgts == action_lgts)[0][0]
                                    temp_obs, reward, done, _ = env.step(action)
                                    temp_obs = np.array([temp_obs])
                                    num += 1

                            if reward == 0:
                                break
                            else:
                                all_obs.append(temp_obs)
                                all_actions.append(int(action))
                                faces_mask[int(action)] = 1

                        for a in all_actions:
                            f.write(str(face_tags[a]))
                            f.write('\t')
                        f.write('\n')


if __name__ == "__main__":
    policy_file = r"Results\WoP_1\ppo\policy.pth"
    part_file = r"Data\evaluation_data\eval"
    custom_env = r"Enviroments/WorldofParts_for_eval"
    action_file = r"Data\evaluation_data\eval_results\_model1_0_action.txt"
    evaluation(policy_file, part_file, custom_env, action_file)