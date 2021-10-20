from gym.envs.registration import register
import gym
from .types import Config
import argparse
parser = argparse.ArgumentParser()

V0NAME = 'GazeboEnv'

parser.add_argument(
    "--map_name",
    type=str,
    choices=[
        "maze1",
        "maze2",
        "maze3",
    ],
    default="maze3",
)

parser.add_argument(
    "--enable_viz_image_cv2",
    dest="viz_image_cv2",
    action="store_true",
    default=True,
)

"""
If True, set the random target position. (Use when training)
If False, set the specific target position. (Use when testing)
"""
parser.add_argument(
    "--random_target",
    default=True,
)

args = parser.parse_args()

if V0NAME not in gym.envs.registry.env_specs:

    register(
        id = 'GazeboEnv-v1',
        entry_point = 'src.gym_gazebo.envs:gazebo_turtlebot3_Env',
        kwargs={'config': Config(
            map_name = args.map_name,
            viz_image_cv2 = args.viz_image_cv2,
            random_target = args.random_target

    )},
        max_episode_steps = 750,
    )
