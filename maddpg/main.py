import numpy as np
import random
import cv2
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
# from make_env import make_env
from smac.env import StarCraft2Env
import datetime
import matplotlib.pyplot as plt
import time


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    # scenario = 'simple'
    # scenario = 'simple_adversary'
    # env = make_env(scenario)

    map_name = "3m_vs_15zg_IM"
    env = StarCraft2Env(map_name=map_name)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    obs_shape = env_info["obs_shape"]
    n_actions = env_info["n_actions"]

    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(obs_shape)
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128,
                           alpha=0.01, beta=0.01,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 20
    N_GAMES = 15000
    MAX_STEPS = env_info["episode_limit"]
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    noise_rate = 0.99
    noise_rate_min = 0.01
    noise_decay_rate = (noise_rate - noise_rate_min) / N_GAMES

    heat_map = None
    hm_size = 10
    name = map_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = cv2.VideoWriter(f"{name}.mp4", -1, 100.0, (320, 320))
    save_every = 100
    start = time.time()

    env.reset()
    state_novelty = np.zeros((env.map_x, env.map_y), dtype=np.uint32)

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        env.reset()
        if heat_map is None:
            heat_map = np.zeros((env.map_x * hm_size,
                                 env.map_y * hm_size))

        obs = env.get_obs()
        score = 0
        done = False
        episode_step = 0
        noise_rate = max(noise_rate_min, noise_rate - noise_decay_rate)
        while not done:
            agents_state_novelties = []
            heat_map *= 0.9999
            positions = np.array(
                [[env.agents[i].health > 0, env.agents[i].pos.x * hm_size, env.agents[i].pos.y * hm_size] for i in
                 env.agents.keys()], dtype=np.uint8)
            heat_map[env.map_y * hm_size - positions[:, 2], positions[:, 1]] = 1

            heatmap_image = np.zeros((env.map_x * hm_size,
                                      env.map_y * hm_size, 3), dtype=np.uint8)
            heatmap_image[:, :, 2] = (heat_map * 255).astype(np.uint8)
            for alive, *pos in positions:
                if alive:
                    cv2.circle(heatmap_image, (pos[0], env.map_y * hm_size - pos[1]), 2,
                               (255, 255, 255), -1)

            if not total_steps % save_every:
                out.write(heatmap_image)

            cv2.imshow("Heatmap", heatmap_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            actions_probabilities = maddpg_agents.choose_action(obs, noise_rate)
            actions = [random.choices(np.arange(n_actions),
                                      weights=actions_probabilities[i] * env.get_avail_actions()[i])[0]
                       for i in range(n_agents)]

            agents_positions = [list(map(int, [env.agents[i].pos.x, env.agents[i].pos.y])) for i in env.agents.keys() if
                                env.agents[i].health > 0]
            for pos in agents_positions:
                state_novelty[pos[0], pos[1]] += 1
                agents_state_novelties.append(state_novelty[pos[0], pos[1]])

            sn_min, sn_max = state_novelty[state_novelty.nonzero()].min(), state_novelty.max()
            intrinsic_rewards = [1 - (x - sn_min) / sn_max for x in agents_state_novelties]

            reward, done, info = env.step(actions)
            print("Reward: ", reward)
            obs_ = env.get_obs()
            # env.agents[0].pos

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = True

            memory.store_transition(obs, state, actions_probabilities, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += reward
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))

    print("End:", (time.time() - start) / 60)
    plt.plot(range(len(score_history)), score_history)
    plt.show()
    out.release()
    cv2.destroyAllWindows()
