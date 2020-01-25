import numpy as np
import tensorflow as tf
import retro
import os
import shutil
import datetime
import cv2
from gym import wrappers
from agent import DQN

# Some control variables
create_video = False
render = False
save_dir = './models/'
save_path = './models/model'

# functions
def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False

    # TODO: change observation to preprocessed pixel inputs!
    observations = env.reset()
    # # print("**************")
    # print("first size")
    # print(len(observations))
    # print("second size")
    # print(len(observations[0]))
    # print("type:")
    # print(type(observations))
    # print("first member:")
    # print(observations[0])
    # print("shape:")
    # print(observations.shape)
    # cv2.imshow("observations image", observations);
    # cv2.waitKey();
    # #print(observations)
    # print("**************")
    # input("pause")

    while not done:
        if render:
            env.render()

        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        # TODO: change observation here too!
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

        # for stable weights
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
    env = retro.make(game='Frogger-Genesis', use_restricted_actions=retro.Actions.DISCRETE)
    env = wrappers.FrameStack(env, 4)
    gamma = 0.99
    copy_step = 25
    num_actions = env.action_space.n
    num_states = len(env.observation_space.sample())
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # For stable weights, use one net to train, and copy their weights over to the TargetNet every copy_steps
    TrainNet = DQN(num_actions=num_actions, gamma=gamma, max_experiences=max_experiences,
                   min_experiences=min_experiences, batch_size=batch_size, lr=lr,
                   hidden_units=hidden_units, num_states=num_states)
    TargetNet = DQN(num_actions=num_actions, gamma=gamma, max_experiences=max_experiences,
                    min_experiences=min_experiences, batch_size=batch_size, lr=lr,
                    hidden_units=hidden_units, num_states=num_states)

    # Loading check
    while True:
        if os.path.exists(save_dir):
            if input("\n\nWould you like to load the previous network weights? (y/n) ") == 'y':
                # load weights and copy to train net
                TargetNet.load_model(save_path)
                TrainNet.copy_weights(TargetNet)
                print("Loaded model weights...")
                break

            elif input("\nWould you like to delete the old checkpoints and start again? (y/n)") == 'y':
                shutil.rmtree(save_dir)
                print("Removed old checkpoint...")
                break
        else:
            break

    N = 50000
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    max_avg_score = 0

    # play N games
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 10):(n + 1)].mean()

        with summary_writer.as_default():
            tf.summary.scalar("episode reward", total_reward, step=n)
            tf.summary.scalar("running avg reward(10)", avg_rewards, step=n)

        if n % 10 == 0:
            if avg_rewards > max_avg_score:
                max_avg_score = avg_rewards

            print("episode:", n, "episode reward:", total_reward,
                  "eps:", epsilon, "avg reward (last 10):", avg_rewards,
                  "max avg reward(10):", max_avg_score)

            # save the model weights
            TargetNet.save_model(save_path)

    print("avg reward for last 10 episodes:", avg_rewards)

    if create_video:
        make_video(env, TrainNet)

    env.close()


if __name__ == '__main__':
    main()
