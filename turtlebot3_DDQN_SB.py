import sys
sys.path.insert(0, '/root/mantis_ws_py3tf/devel/lib/python3/dist-packages')
sys.path.insert(1, '/root/mantis_ws_py3tf/src/mantis_ddqn_navigation')
# sys.path.append('/root/mantis_ws_py3tf/src/mantis_ddqn_navigation')
# sys.path.append('/root/mantis_ws_py3tf/devel/lib/python3/dist-packages')
print(sys.path)
import gym
import src.gym_gazebo.envs
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import pickle
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

from stable_baselines import DDPG, ACKTR, PPO2, DQN
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.buffers import PrioritizedReplayBuffer
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback

from CustomDQNModel import FeedForwardPolicy
# from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # tf.config.experimental.set_virtual_device_configuration(gpus[0],
            # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*18)])
            tf.config.experimental.set_memory_growth(gpu, True)                     # Dynamic allocate GPU memory
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='GazeboEnv-v1')
parser.add_argument('--weights', type=str, default=None)

args = parser.parse_args()

# Create dir
moniter_dir = "./DQN_model/moniter/"
os.makedirs(moniter_dir, exist_ok=True)

final_model_dir = "./DQN_model/final_model/"
os.makedirs(final_model_dir, exist_ok=True)

if args.mode == 'train':
    train = True
elif args.mode == 'test':
    train = False

# Custom MLP policy of three layers of size 128 each
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 64],
                                           act_fun=tf.nn.relu,
                                           dueling = True,
                                           feature_extraction="mlp")

#互動環境之id
env_id = args.env_name
#使用幾個執行緒multi-thread
num_cpu = 1
#深度學習演算法
ALGO = DQN
#總訓練步數
TRAIN_STEPS = 2400000
# Number of episodes for evaluation
EVAL_EPS = 10

model_index = 1200000
# log_filename = './DQN_model/Checkpoint/' + 'DQN_{}_steps.zip'.format(model_index)
log_filename = './DQN_model/moniter/' + 'best_model_1.zip'

if __name__ == '__main__':

    if train:

        # 將互動環境向量化(vecterized environment)，使訓練過程效率更高。
        # env = make_vec_env(env_id, n_envs=num_cpu, seed=123, vec_env_cls=SubprocVecEnv)
        # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
        env = gym.make(env_id)
        env = Monitor(env, moniter_dir)

        # We create a separate environment for evaluation
        eval_env = DummyVecEnv([lambda: gym.make(env_id)])
        # eval_env = gym.make(env_id)

        # np.random.seed(123)
        # env.seed(123)
        # check_env(env)

        # 動作數量與動作雜訊
        n_actions = env.action_space.n
        param_noise = None
        # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        # policy為神經網路
        # env為向量化互動環境
        # verbose為顯示訓練進度
        # tensorboard_log為訓練過程tensorboard存放位置
        # n_cpu_tf_sess為1代表completely deterministic
        # exploration_fraction=0.25
        model = ALGO(policy=CustomDQNPolicy, env=env, gamma=0.995, learning_rate=0.00025, buffer_size=120000, exploration_fraction=0.25,
                 exploration_final_eps=0.05, exploration_initial_eps=1, train_freq=3, batch_size=64, double_q=True,
                 learning_starts=5000, target_network_update_freq=2000, prioritized_replay=True, prioritized_replay_alpha=0.6,
                 prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-5,
                 verbose=1, tensorboard_log='./DQN_model/tensorboard/', full_tensorboard_log=True, n_cpu_tf_sess=1)

        # model = SAC(CustomSACPolicy, env, buffer_size=100000, learning_starts=5000, verbose=1, batch_size=64,
        # action_noise=action_noise, random_exploration=0.3, tensorboard_log='./SAC_tensorboard/')
        # model = ALGO(policy=CustomPPO2Policy, env=env, n_steps=128, learning_rate=1e-2, verbose=1, tensorboard_log='./GOD_model/tensorboard/')


        # 每5000步定期儲存模型callback
        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./DQN_model/Checkpoint/',
                                                 name_prefix='DQN')

        # 每5000步定期執行evaluate，則儲存reward最佳的模型
        # save_best_reward_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=moniter_dir)
        save_best_reward_model_callback = EvalCallback(eval_env, best_model_save_path=moniter_dir, log_path=moniter_dir,
                                                       n_eval_episodes=EVAL_EPS, eval_freq=5000, deterministic=True,
                                                       render=True, verbose=1)

        # Create the callback list
        callback = CallbackList([checkpoint_callback, save_best_reward_model_callback])
        # callback = CallbackList([checkpoint_callback])

        print("Load model from : {}".format(log_filename))
        # 若存在指定的模型，則載入模型繼續訓練
        if os.path.isfile(log_filename):
            model = ALGO.load(log_filename, exploration_final_eps=0.05, exploration_initial_eps=0.05,
                              tensorboard_log='./DQN_model/tensorboard/', full_tensorboard_log=True)
            model.set_env(env)
            print("Weight loaded!!!")
            model.learn(total_timesteps=TRAIN_STEPS, log_interval=1, reset_num_timesteps=False, callback=callback)
        # 若不存在則重新訓練
        else:
            print("Weight not found!!!")
            model.learn(total_timesteps=TRAIN_STEPS, log_interval=1, reset_num_timesteps=True, callback=callback)

        # 儲存最終模型，但不一定是最佳
        model.save(final_model_dir + "DQN_maze3")
        # 繪出訓練過程的學習曲線
        plot_results(moniter_dir)
        del model

        # Close the processes
        env.close()
    # Test Part 載入模型進行測試
    else:

        env = DummyVecEnv([lambda: gym.make(env_id)])
        evaluate_times = 10

        for i in range(710000, 720001, 10000):

            # Best Model
            # log_filename = './DQN_model/moniter/' + 'best_model.zip'

            # Checkpoint Model
            log_filename = './DQN_model/Checkpoint/' + 'DQN_{}_steps.zip'.format(i)

            # Final Model
            # log_filename = './DQN_model/final_model/' + 'DQN_maze3.zip'

            if os.path.isfile(log_filename):
                model = ALGO.load(log_filename)
                model.set_env(env)
                print("log_filename:" + log_filename)
                print("Weight loaded!!!")

            obs = env.reset()
            # with open('./DQN_model' + '/validation.txt', 'a') as fp:
            #     fp.write("Test {} model:\n".format(log_filename))

            for test_time_per_model in range(evaluate_times):
                while 1:
                    action, _states = model.predict(obs)
                    obs, rewards, dones, info = env.step(action)
                    if dones == True:
                        break