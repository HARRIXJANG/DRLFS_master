from Enviroments.envs import EnvUnits
import gym
import numpy as np
import random
from gym import spaces
import torch as th
import copy

class WorldofPartsEnv(gym.Env):
    # mode : 'PNAConv', 'EGATConv', 'GINEConv'
    # env_mode : 'with_BF', 'without_BF'
    def __init__(self, path_of_part, sphere, graphs, max_bfs_normals, base_face_ids, processing_strategy=0,
                 rough_finish=1, action_size=50, max_step=100, device='cuda', max_repeat_step = 8):
        self.path_of_part = path_of_part
        self.processing_strategy = processing_strategy
        self.rough_finish = rough_finish
        self.action_size = action_size
        self.max_step = max_step
        self.device = device
        self.tolerance = 45

        self.step_num = 0
        self.first_step = 0
        self.max_repeat_step = max_repeat_step

        self.graphs = graphs
        self.max_bfs_normals = max_bfs_normals
        self.base_face_ids = base_face_ids

        self.temp_all_faces_ep1 = []
        self.Temp_SP = th.tensor(sphere)

        self.observation_space = spaces.Graph(
            node_space=spaces.Box(low=-1, high=1, shape=(len(self.graphs[0].dstdata["NodeAttr"][0]), 1)),
            edge_space=spaces.Discrete(2))

        self.action_space = spaces.MultiBinary(self.action_size)

    def _get_obs(self):
        return copy.deepcopy(self._selected_faces_graph).to(self.device)

    def _get_info(self):
        return {}

    def reset(self, seed=None, return_info=False, options=None):
        super(WorldofPartsEnv, self).reset(seed=seed)
        GraphID = random.randint(0, len(self.graphs) - 1)
        self.temp_graph = self.graphs[GraphID].to(self.device)
        self.num_of_nodes = self.temp_graph.num_nodes()

        if th.sum(self.temp_graph.dstdata["NodeAttr"][:, -2]) >= 1:
            self.wether_base_face = 1
        else:
            self.wether_base_face = 0
        self.temp_base_face_ids = th.tensor(self.base_face_ids[GraphID]).to(self.device)
        self.max_bfs_normal = th.tensor(self.max_bfs_normals[GraphID]).to(self.device)


        self.step_num = 0
        self.temp_all_faces_ep1.clear()

        self._selected_faces = th.tensor(np.zeros(self.num_of_nodes)).to(self.device)
        self.first_step = 1

        self._selected_faces_graph = self.temp_graph
        self._selected_faces_graph.dstdata["NodeAttr"][:, -1] = th.zeros(self._selected_faces_graph.dstdata["NodeAttr"].shape[0])

        observation = self._get_obs()
        info = self._get_info()

        return (observation, info) if return_info else observation

    def step(self, action):
        if (action >= self.temp_graph.num_nodes()):
            reward = 0
        else:
            acs = self.one_hot(action, self.action_size)
            temp_selected_faces = self._selected_faces + acs[:self.num_of_nodes]
            temp_selected_faces_graph = copy.deepcopy(self._selected_faces_graph).to(self.device)
            temp_selected_faces_graph.dstdata["NodeAttr"][:, -1] += th.tensor(acs[:self.num_of_nodes])

            temp_selected_id = th.where(acs == 1)[0][0]
            indexs_state = th.where(self._selected_faces == 1)[0]

            if self.first_step == 1:
                if self.wether_base_face == 1:
                    if self.temp_graph.dstdata["NodeAttr"][temp_selected_id, -2] == 1:
                        reward = self.base_face_reward(temp_selected_id, 5)
                        self.base_face_id = temp_selected_id
                        self.first_step = 0
                        self._selected_faces = temp_selected_faces
                        self._selected_faces_graph = temp_selected_faces_graph
                        self.temp_all_faces_ep1.append(temp_selected_id)
                        self.base_face_center_point = th.tensor(self.temp_graph.dstdata["NodeAttr"][self.base_face_id, 9:12])
                        self.base_face_normal = th.tensor(self.temp_graph.dstdata["NodeAttr"][self.base_face_id, 12:15])
                    else:
                        reward = 0
                else:
                    reward = 5
                    self.first_step = 0
                    self._selected_faces = temp_selected_faces
                    self._selected_faces_graph = temp_selected_faces_graph
                    self.temp_all_faces_ep1.append(temp_selected_id)
            else:
                if self._selected_faces[temp_selected_id] == 1:
                    reward = 0
                else:
                    temp_all_faces_ep1_copy = copy.deepcopy(self.temp_all_faces_ep1)
                    temp_all_faces_ep1_copy.append(temp_selected_id)
                    all_face_normals = th.tensor(self.temp_graph.dstdata["NodeAttr"][temp_all_faces_ep1_copy, 12:15])
                    TD = EnvUnits.Tool_Direction(all_face_normals, self.Temp_SP, self.device)
                    cover = TD.judge_tool_cover()
                    if cover == 0:
                        reward = 0
                    else:
                        if self.wether_base_face == 1:
                            if self.judge_neighboor(self.base_face_id, temp_selected_id) == 1:
                                if self.judge_concave_tensor(th.tensor(self.base_face_id), th.tensor(temp_selected_id)) == 1:
                                    reward = 1
                                    self._selected_faces = temp_selected_faces
                                    self._selected_faces_graph = temp_selected_faces_graph
                                    self.temp_all_faces_ep1.append(temp_selected_id)
                                else:
                                    reward = 0
                            else:
                                index_actions = th.tile(temp_selected_id, [indexs_state.shape[0]]).to(self.device)
                                jn = self.temp_graph.has_edge_between(indexs_state, index_actions)

                                if (jn == False).all():
                                    reward = 0
                                else:
                                    face_center_point = th.tensor(self.temp_graph.dstdata["NodeAttr"][temp_selected_id, 9:12])
                                    above_base_face = self.JPRB(face_center_point)
                                    if above_base_face == 1:
                                        face_bf = th.tensor(self.temp_graph.dstdata["NodeAttr"][temp_selected_id, -2])
                                        face_vector = th.tensor(self.temp_graph.dstdata["NodeAttr"][temp_selected_id, 12:15])
                                        angle = self.get_angle(face_vector)
                                        if (angle <= self.tolerance)&(face_bf == 1):
                                            reward = 0
                                        else:
                                            if self.judge_required_faces(temp_selected_id) == 1:
                                                reward = 1
                                                self._selected_faces = temp_selected_faces
                                                self._selected_faces_graph = temp_selected_faces_graph
                                                self.temp_all_faces_ep1.append(temp_selected_id)
                                            else:
                                                reward = 0
                                    else:
                                        reward = 0
                        else:
                            tensor_temp_all_faces_ep1 = th.tensor(self.temp_all_faces_ep1).to(self.device)
                            temp_selected_ids = th.tile(temp_selected_id, [tensor_temp_all_faces_ep1.shape[0]])
                            flag = self.judge_concave_tensor(tensor_temp_all_faces_ep1, temp_selected_ids)
                            if flag == 0:
                                reward = 0
                            else:
                                reward = 1
                                self.temp_all_faces_ep1.append(temp_selected_id)
                                self._selected_faces = temp_selected_faces
                                self._selected_faces_graph = temp_selected_faces_graph

        done = 0
        if reward == 0:
            if self.first_step == 0:
                self.step_num += 1
                if self.step_num >= self.max_repeat_step:
                    done = 1
                    self.step_num = 0
                    self.temp_all_faces_ep1.clear()
        else:
            self.step_num = 0

        if sum(self._selected_faces) >= self.num_of_nodes:
            done = 1
            self.step_num = 0
            self.temp_all_faces_ep1.clear()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def one_hot(self, num, size):
        o_h = np.zeros(size)
        o_h[num] = 1
        return th.tensor(o_h).to(self.device)

    def judge_neighboor(self, node1_id, node2_id):
        node1_neighboors = self.temp_graph.in_edges(node1_id)[0]
        index = th.where(node1_neighboors == node2_id)[0]
        if len(index) == 0:
            flag = 0
        else:
            flag = 1
        return flag

    def judge_concave_all(self, node_id):
        node_neighboors = self.temp_graph.in_edges(node_id)[0]
        edge_ids = self.temp_graph.edge_ids(th.ones(node_neighboors.shape[0], dtype=int).to(self.device) * int(node_id),
                                            th.tensor(node_neighboors).to(self.device))
        edge_attr = th.tensor(self.temp_graph.edata["EdgeAttr"][edge_ids]).to(self.device)
        edge_concave = th.tensor(np.tile(np.array([1, 0]), (edge_ids.shape[0], 1))).to(self.device)
        if (edge_attr != edge_concave).all():
            flag = 0
        else:
            flag = 1

        return flag

    def judge_concave_tensor(self, node1_ids, node2_ids):
        jn = self.temp_graph.has_edge_between(node1_ids, node2_ids)
        if jn.all() == np.zeros(jn.shape[0]):
            flag = 0
        else:
            node1_ids_mask = th.masked_select(node1_ids, jn)
            node2_ids_mask = th.masked_select(node2_ids, jn)
            edge_ids = self.temp_graph.edge_ids(node1_ids_mask, node2_ids_mask)
            edge_attr = th.tensor(self.temp_graph.edata["EdgeAttr"][edge_ids[:]])
            if (edge_attr == th.tile(th.tensor([1, 0]), [edge_ids.shape[0], 1]).to(self.device)).all():
                flag = 1
            else:
                flag = 0
        return flag

    def JPRB(self, face_center_point):
        u = face_center_point - self.base_face_center_point
        v = self.base_face_normal
        pro = th.sum(u*v)/th.sum(v*v)
        if pro >= 0:
            return 1
        else:
            return 0

    def get_angle(self, face_normal):
        a = th.sum(face_normal * self.base_face_normal, dim=-1)
        angle = 180*th.acos(a)/th.pi
        return angle


    def judge_required_faces(self, node_id):
        node_neighboors = self.temp_graph.in_edges(node_id)[0]
        edge_ids = self.temp_graph.edge_ids(th.ones(node_neighboors.shape[0], dtype=int).to(self.device) * int(node_id),
                                            th.tensor(node_neighboors).to(self.device))
        edge_attr = th.tensor(self.temp_graph.edata["EdgeAttr"][edge_ids]).to(self.device)

        edge_concave = th.tensor(th.tile(th.tensor([1, 0]), (edge_ids.shape[0], 1))).to(self.device)
        concave_mask = edge_attr[:,0] == edge_concave[:,0]
        concave_indexs = th.where(concave_mask == True)[0]
        concave_neighboors = node_neighboors[concave_indexs]
        wether_base_faces = th.tensor(self.temp_graph.dstdata["NodeAttr"][concave_neighboors, -2]).to(self.device)

        faces_normals = th.tensor(self.temp_graph.dstdata["NodeAttr"][concave_neighboors, 12:15]).to(self.device)
        angles = self.get_angle(faces_normals)
        judge_angles = angles <= self.tolerance
        judge_angles = judge_angles.long()

        t_wether_base_faces = th.tile(th.tensor([2]), [wether_base_faces.shape[0]]).to(self.device)
        if ((wether_base_faces+judge_angles) == t_wether_base_faces).any():
            return 0
        else:
            return 1

    def base_face_reward(self, face_id, reward):
        if (self.max_bfs_normal == self.temp_graph.dstdata["NodeAttr"][face_id, 12:15]).all():
            temp_reward = reward
        else:
            temp_reward = 1
        return temp_reward

class WorldofPartsEnv_for_eval(WorldofPartsEnv):
    def __init__(self, path_of_part, sphere, graphs, max_bfs_normals, base_face_ids, processing_strategy=0,
                 rough_finish=1, action_size=50, max_step=100, device='cuda', max_repeat_step=8):
        super(WorldofPartsEnv_for_eval, self).__init__(path_of_part, sphere, graphs, max_bfs_normals, base_face_ids,
                                                         processing_strategy,
                                                         rough_finish, action_size, max_step, device, max_repeat_step)

    def assignment_id(self, graph_id):
        self.graph_id = graph_id

    def reset(self, seed=None, return_info=False, options=None):
        super(WorldofPartsEnv_for_eval, self).reset(seed=seed)
        GraphID = self.graph_id
        self.temp_graph = self.graphs[GraphID].to(self.device)
        self.num_of_nodes = self.temp_graph.num_nodes()

        if th.sum(self.temp_graph.dstdata["NodeAttr"][:, -2]) >= 1:
            self.wether_base_face = 1
        else:
            self.wether_base_face = 0
        self.temp_base_face_ids = th.tensor(self.base_face_ids[GraphID]).to(self.device)
        self.max_bfs_normal = th.tensor(self.max_bfs_normals[GraphID]).to(self.device)

        self.step_num = 0
        self.temp_all_faces_ep1.clear()

        self._selected_faces = th.tensor(np.zeros(self.num_of_nodes)).to(self.device)
        self.first_step = 1

        self._selected_faces_graph = self.temp_graph
        self._selected_faces_graph.dstdata["NodeAttr"][:, -1] = th.zeros(
            self._selected_faces_graph.dstdata["NodeAttr"].shape[0])

        observation = self._get_obs()
        info = self._get_info()

        return (observation, info) if return_info else observation