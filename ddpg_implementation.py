import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt


# simple feedforward neural net
def ANN(x, layer_sizes, hidden_activation=tf.nn.relu, output_activation=None):
    for h in layer_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=hidden_activation)
    return tf.layers.dense(x, units=layer_sizes[-1], activation=output_activation)


def create_networks(s, a, num_actions, actions_max, hidden_sizes=(300,)):
    hidden_sizes = list(hidden_sizes)
    with tf.variable_scope("mu"):
        mu = actions_max * ANN(s, hidden_sizes + [num_actions], output_activation=tf.tanh)
    with tf.variable_scope("q"):
        inp = tf.concat([s, a], -1)
        q = ANN(inp, hidden_sizes + [1])
        q = tf.squeeze(q, axis=1)
    with tf.variable_scope("q", reuse=True):
        inp = tf.concat([s, mu], -1)
        q_mu = ANN(inp, hidden_sizes + [1])
        q_mu = tf.squeeze(q_mu, axis=1)

    return mu, q, q_mu


# get all variables within a scope
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


class Replay_Buffer(object):
    def __init__(self, batch_size, max_size, num_observations, num_actions):
        self.states = np.empty(shape=[max_size, num_observations], dtype=np.float32)
        self.actions = np.empty(shape=[max_size, num_actions], dtype=np.float32)
        self.rewards = np.empty(shape=[max_size], dtype=np.float32)
        self.next_states = np.empty(shape=[max_size, num_observations], dtype=np.float32)
        self.dones = np.empty(shape=[max_size], dtype=np.int32)

        self.out_states = np.empty(shape=[batch_size, num_observations], dtype=np.float32)
        self.out_actions = np.empty(shape=[batch_size, num_actions], dtype=np.float32)
        self.out_rewards = np.empty(shape=[batch_size], dtype=np.float32)
        self.out_next_states = np.empty(shape=[batch_size, num_observations], dtype=np.float32)
        self.out_dones = np.empty(shape=[batch_size], dtype=np.int32)

        self.batch_size = batch_size
        self.max_size = max_size

        self.pointer = 0
        self.size = 0

    def add(self, s, a, r, ns, d):
        self.states[self.pointer] = s
        self.actions[self.pointer] = a
        self.rewards[self.pointer] = r
        self.next_states[self.pointer] = ns
        self.dones[self.pointer] = d

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = np.random.randint(low=0, high=len(self), size=self.batch_size)
        self.out_states = self.states[idxs].copy()
        self.out_actions = self.actions[idxs].copy()
        self.out_rewards = self.rewards[idxs].copy()
        self.out_next_states = self.next_states[idxs].copy()
        self.out_dones = self.dones[idxs].copy()

        return (
            self.out_states,
            self.out_actions,
            self.out_rewards,
            self.out_next_states,
            self.out_dones
        )

    def __len__(self):
        return self.size


class DDPG(object):
    def __init__(
            self,
            sess,
            num_observations,
            num_actions,
            actions_max,
            mu_lr,
            q_lr,
            tau,
            noise_magn,
            gamma):

        self.sess = sess
        self.num_actions = num_actions
        self.actions_max = actions_max
        self.noise_magn = noise_magn
        self.gamma = gamma

        self.states = tf.placeholder(shape=[None, num_observations], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
        self.G = tf.placeholder(shape=[None], dtype=tf.float32)

        with tf.variable_scope("main"):
            self.mu, self.q, self.q_mu = create_networks(
                self.states, self.actions, num_actions, actions_max)
        with tf.variable_scope("target"):
            _, _, self.q_mu_targ = create_networks(
                self.states, self.actions, num_actions, actions_max)

        self.mu_loss = - tf.reduce_sum(self.q_mu)

        self.q_loss = tf.reduce_sum(tf.square(self.G - self.q))

        mu_optimizer = tf.train.AdamOptimizer(mu_lr)
        q_optimizer = tf.train.AdamOptimizer(q_lr)
        self.mu_train_op = mu_optimizer.minimize(self.mu_loss, var_list=get_vars("main/mu"))
        self.q_train_op = q_optimizer.minimize(self.q_loss, var_list=get_vars("main/q"))

        self.hard_update = tf.group(
            [p.assign(q) for p, q in zip(get_vars("target"), get_vars("main"))]
        )

        self.soft_update = tf.group(
            [p.assign(tau * p + (1 - tau) * q)
             for p, q in zip(get_vars("target"), get_vars("main"))]
        )

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.hard_update)

    def get_action(self, state):
        state = np.atleast_2d(state)
        action = self.sess.run(self.mu, feed_dict={
            self.states: state
        }).squeeze()
        noise = self.noise_magn * np.random.randn(self.num_actions).squeeze()
        action += noise
        action = np.clip(action, -self.actions_max, self.actions_max)
        action = np.atleast_1d(action)
        return action

    def get_action_deterministic(self, state):
        state = np.atleast_2d(state)
        action = self.sess.run(self.mu, feed_dict={
            self.states: state
        }).squeeze()
        action = np.atleast_1d(action)
        return action

    def get_q_mu_target(self, state):
        return self.sess.run(self.q_mu_targ, feed_dict={
            self.states: state
        })

    def train(self, batch):
        s, a, r, ns, d = batch
        G = r + (1 - d) * self.gamma * self.get_q_mu_target(ns)
        self.sess.run([self.mu_train_op, self.q_train_op, self.soft_update], feed_dict={
            self.states: s,
            self.actions: a,
            self.G: G
        })


if __name__ == "__main__":

    gamma = 0.99
    n_episodes = 600
    test_interval = 25
    max_replay_size = 1000000
    tau = 0.995
    mu_lr = 1e-3
    q_lr = 1e-3
    batch_size = 100
    start_steps = 10000
    noise_magn = 0.15
    max_episode_length = 1000
    env = gym.make("Ant-v3")
    #env = gym.wrappers.Monitor(env, "ddpg_implementation_monitor/")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0]
    sess = tf.Session()

    agent = DDPG(
        sess=sess,
        num_observations=obs_dim,
        num_actions=act_dim,
        actions_max=action_max,
        mu_lr=mu_lr,
        q_lr=q_lr,
        tau=tau,
        noise_magn=noise_magn,
        gamma=gamma
    )
    rb = Replay_Buffer(
        batch_size=batch_size,
        max_size=max_replay_size,
        num_observations=obs_dim,
        num_actions=act_dim
    )

    num_global_steps = 0
    average_epoch_reward = np.empty(n_episodes)
    for e in range(n_episodes):
        print(f"Epoch  {e}, Global step {num_global_steps}, Mean Reward: {average_epoch_reward[max(0, e-1)]}")
        d = False
        steps = 0
        s = env.reset()
        cumulative_epsiode_reward = 0
        while not d:
            if num_global_steps < start_steps:
                a = env.action_space.sample()
            else:
                a = agent.get_action(s)
            ns, r, d, _ = env.step(a)
            rb.add(s, a, r, ns, d)
            cumulative_epsiode_reward += r
            steps += 1
            num_global_steps += 1
            s = ns
        average_epoch_reward[e] = cumulative_epsiode_reward
        if len(rb) < batch_size:
            continue
        for _ in range(steps):
            batch = rb.sample()
            agent.train(batch)

    cummean = average_epoch_reward.cumsum() / np.arange(n_episodes)
    plt.plot(average_epoch_reward.cumsum() / np.arange(n_episodes))
    plt.show()

    d = False
    s = env.reset()
    while not d:
        env.render()
        a = agent.get_action_deterministic(s)
        ns, r, d, _ = env.step(a)
        s = ns
