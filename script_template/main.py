import numpy as np
import tensorflow as tf
import retro
import os
import datetime
from gym import wrappers
from agent import DQN

# Some control variables
create_video = False
render = False

# functions
def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False

    # TODO: chanbe observations below to use the pixel values for the CNN agent. Leave as is for gamestate agent I guess?
    observations = env.reset()

    while not done:
        if render:
            env.render()

        # TODO: change observations here as well!
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations

        observations, reward, done, _ = env.step(action)
        rewards += reward

        if done:
            reward = -200
            env.reset()

        # Add experience to replay bank
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)
        iter += 1

        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards

def make_video(env, TrainNet):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()

    while not done:
        action = TrainNet.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ").format(steps, rewards)

def main():
    env = retro.make(game='Frogger-Genesis')
    gamma = 0.99
    copy_step = 25
    num_actions = env.action_space.n
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now()
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_action=num_actions, gamma=gamma, max_experiences=max_experiences,
                   min_experiences=min_experiences, batch_size=batch_size, lr=lr)
    TargetNet = DQN(num_action=num_actions, gamma=gamma, max_experiences=max_experiences,
                   min_experiences=min_experiences, batch_size=batch_size, lr=lr)
    N = 50000
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1

    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()

        with summary_writer.as_default():
            tf.summary.scalar("episode reward", total_reward, step=n)
            tf.summary.scalar("running avg reward(100)", avg_rewards, step=n)

        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards)

    print("avg reward for last 100 episodes:", avg_rewards)

    if create_video:
        make_video(env, TrainNet)

    env.close()

if __name__ == '__main__':
    main()