import gym
import torch as th
import MainUnits
import os
from torch.distributions import Categorical
import Enviroments
from Enviroments.envs import EnvUnits
import tianshou as ts
import NetworkModel
import time
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger



def construct_envs(custom_env, path_of_part_train, path_of_part_test, max_repeat_step = 8):
    SP = MainUnits.SpherePoints()
    spherepoints = SP.discretize_sphere()
    graphs_train, max_bfs_normals_train, base_face_ids_train = EnvUnits.ReadDataFromPath_v2(path_of_part_train,
                                                                                            Mode="PNAConv",
                                                                                            EnvMode="without_BF")
    graphs_test, max_bfs_normals_test, base_face_ids_test = EnvUnits.ReadDataFromPath_v2(path_of_part_test,
                                                                                            Mode="PNAConv",
                                                                                            EnvMode="without_BF")
    env = gym.make(custom_env,
                   path_of_part=path_of_part_train,
                   sphere=spherepoints,
                   graphs=graphs_train,
                   max_bfs_normals=max_bfs_normals_train,
                   base_face_ids=base_face_ids_train,
                   )

    train_envs = ts.env.DummyVectorEnv([lambda: gym.wrappers.AutoResetWrapper(
        gym.make(custom_env,
                 path_of_part=path_of_part_train,
                 sphere=spherepoints,
                 graphs = graphs_train,
                 max_bfs_normals = max_bfs_normals_train,
                 base_face_ids = base_face_ids_train,
                 max_repeat_step = max_repeat_step))] * 400)

    test_envs = ts.env.DummyVectorEnv([lambda: gym.wrappers.AutoResetWrapper(
        gym.make(custom_env,
                 path_of_part=path_of_part_test,
                 sphere=spherepoints,
                 graphs=graphs_test,
                 max_bfs_normals=max_bfs_normals_test,
                 base_face_ids=base_face_ids_test,
                 max_repeat_step = max_repeat_step))] * 700)

    print("max_repeat_step: "+ str(max_repeat_step))
    return env, train_envs, test_envs

def construct_policy_PPO(env):
    in_node_feats = env.observation_space.node_space.shape[0]
    action_feats = env.action_space.shape[0]
    Actor_net = NetworkModel.Actor_Net_PNAConv_Model(in_node_feats, action_feats).to("cuda")
    Critic_net = NetworkModel.Critic_Net_PNAConv_Model(in_node_feats, action_feats).to("cuda")
    optim = th.optim.Adam([{'params':Actor_net.parameters()}, {'params': Critic_net.parameters()}], lr=3e-4)

    dist_fn = th.distributions.Categorical
    policy = ts.policy.PPOPolicy(actor=Actor_net, critic=Critic_net, optim = optim, dist_fn=dist_fn)
    return policy

def train_PPO():
    max_repeat_step = 20
    env, train_envs, test_envs = construct_envs("Enviroments/WorldofParts",
                                                r"Data\train_data",
                                                r"Data\test_data",
                                                max_repeat_step
                                                )

    train_envs.seed(1998)
    test_envs.seed(1998)

    policy = construct_policy_PPO(env)

    time_start = time.time()
    epoch = 20
    step_per_epoch = 20000
    repeat_per_collect = 5
    batch_size = 512
    test_num = 700
    buffer_size = 40000
    step_per_collect = 4000

    print("repeat_per_collect: " + str(repeat_per_collect))
    print("batch_size: " + str(batch_size))
    print("PPO")

    train_collector = ts.data.Collector(policy=policy, env=train_envs,
                                        buffer=ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=train_envs.env_num),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy=policy, env=test_envs, exploration_noise=True)


    logdir = "Results"
    task = "WoP"

    print(task)

    log_path = os.path.join(logdir, task, 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # trainer
    def save_best_fn(policy):
        th.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


    result = ts.trainer.onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        epoch,
        step_per_epoch,
        repeat_per_collect,
        test_num,
        batch_size,
        step_per_collect= step_per_collect,
        save_best_fn=save_best_fn,
        logger=logger
    )
    time_end = time.time()
    print(str(time_end - time_start))
    return result

if __name__ == "__main__":
    train_PPO()
