import warnings
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.policies import BasePolicy, nature_cnn, mlp_extractor, register_policy, ActorCriticPolicy
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.tf_layers import mlp
from stable_baselines.deepq.policies import DQNPolicy


from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, ELU, Reshape, ReLU
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Flatten, Add, Lambda, Concatenate, LayerNormalization

IMG_H = 48
IMG_W = 96
IMG_C = 1
VECTOR_LEN = 364


class FeedForwardPolicy(DQNPolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                # ?????????observation????????????????????????CNN?????????????????????
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    obs_feature = self._feature_extract(self.processed_obs, reuse=reuse)
                    extracted_features = tf.layers.flatten(obs_feature)
                    action_out = extracted_features
                    for layer_size in layers:
                        action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                        action_out = act_fun(action_out)

                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        self.q_values = q_out
        self._setup_init()

    def _feature_extract(self, obs=None, reuse=True, scope="feature"):
        """
        ???????????????????????????1D?????????????????????????????????1D???????????????????????????????????????????????????????????????????????????1D???????????????48*96????????????364????????????
        ??????48*96??????????????????(48, 96, 1)?????????????????????CNN?????????????????????128??????????????????1
        364????????????360??????????????????????????????????????????????????????60?????????????????????4????????????????????????????????????64?????????
        ?????????64???????????????????????????????????????????????????128+64????????????192????????????????????????????????????11???????????????
        :param obs: observation???4972??????????????????4608??????(48, 96, 1)????????????364??????????????????????????????????????????????????????
        :param reuse: (bool) If the policy is reusable or not
        :param scope:
        :return: ?????????????????????Observation????????????128+128??????
        """
        activ = tf.nn.relu

        img_vec, infos_input = tf.split(obs, num_or_size_splits=[IMG_H * IMG_W * IMG_C, VECTOR_LEN], axis=1)

        # with tf.variable_scope(scope, reuse=reuse):
        "extract the feature from img"
        img = tf.keras.layers.Reshape((IMG_H, IMG_W, IMG_C))(img_vec)
        img_c1 = activ(conv(img, 'c1', n_filters=32, filter_size=4, stride=4, init_scale=np.sqrt(2)))
        img_c2 = activ(conv(img_c1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2)))
        img_c3 = activ(conv(img_c2, 'c3', n_filters=64, filter_size=1, stride=1, init_scale=np.sqrt(2)))
        img_flatten = conv_to_fc(img_c3)
        img_fc1 = activ(linear(img_flatten, 'img_fc1', n_hidden=128, init_scale=np.sqrt(2)))
        # img_fc2 = activ(linear(img_fc1, 'img_fc2', n_hidden=256, init_scale=np.sqrt(2)))

        "extract the feature from 1D vector"
        lidar_infos, target_infos = tf.split(infos_input, num_or_size_splits=[360, 4], axis=1)
        lidar_infos = Flatten()(lidar_infos)
        target_infos = Flatten()(target_infos)

        lidar_d1 = activ(linear(lidar_infos, 'lidar_fc1', n_hidden=180, init_scale=np.sqrt(2)))
        lidar_d2 = activ(linear(lidar_d1, 'lidar_fc2', n_hidden=60, init_scale=np.sqrt(2)))

        "Concatenate"
        infos_feature = tf.concat([lidar_d2, target_infos], axis=-1) # 60 + 4
        obs_feature = tf.concat([img_fc1, infos_feature], axis=-1)      # 128 + 64
        return obs_feature

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
