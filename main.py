import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ddpg_tensorflow.ddpg import DDPG, Replay_Buffer

# === Hyperparams ===
gamma = 0.99
n_episodes = 200
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
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]

# Soft assignment of GPU resource => https://github.com/tensorflow/tensorflow/issues/1578
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# === Instantiate Agent/ER ===
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

# === Training Phase ===
num_global_steps = 0
average_epoch_reward = np.empty(n_episodes)
for e in range(n_episodes):
    print(f"Epoch  {e}, Global step {num_global_steps}, Mean Reward: {average_epoch_reward[max(0, e - 1)]}")
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

# === Data visualisation ===
cummean = average_epoch_reward.cumsum() / np.arange(n_episodes)
plt.plot(average_epoch_reward.cumsum() / np.arange(n_episodes))
plt.xlabel("episode")
plt.ylabel("MAR")
plt.title("Moving Average Reward Over Episodes")
plt.axes()
plt.grid()
plt.savefig("./images/mar.png")

# Deploy the trained agent on the env to see how well it's learnt
d = False
s = env.reset()
while not d:
    env.render()
    a = agent.get_action_deterministic(s)
    ns, r, d, _ = env.step(a)
    s = ns
