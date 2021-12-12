import numpy as np
import random
import cv2
import os
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from smac.env import StarCraft2Env
import datetime
import time
from torch.utils.tensorboard import SummaryWriter


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    map_name = "2m_vs_10zg_IM"
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
                           alpha=0.0005, beta=0.0005, chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(10_000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 1000
    N_STEPS = 200_000
    MAX_STEPS = env_info["episode_limit"]
    score_history = []
    ep_len_history = []
    evaluate = False
    best_score = 0

    noise = np.linspace(1, 0, num=N_STEPS)
    a = np.linspace(1, 3, num=N_STEPS)
    x = np.arange(N_STEPS)
    y = (np.cos(2 * np.pi * 4.5 * x / N_STEPS) + 1) * a
    sin_noise = (noise + noise * y) / 3 - 0.01

    # noise_rate = 0.99
    noise_rate_min = 0.01
    # noise_decay_rate = noise_rate / N_STEPS

    heat_map = None
    hm_size = 10

    time_date = datetime.datetime.now().strftime('%Y_%m_%d')
    time_day = datetime.datetime.now().strftime('%H_%M_%S')
    train_info_folder = "/train_info/"

    folder_name = train_info_folder + time_date + "/" + time_day + "/"

    if not os.path.isdir(os.getcwd() + train_info_folder + time_date):
        os.mkdir(os.getcwd() + train_info_folder + time_date)
    if not os.path.isdir(os.getcwd() + folder_name):
        os.mkdir(os.getcwd() + folder_name)

    writer = SummaryWriter(os.getcwd() + folder_name + "logs")

    out = cv2.VideoWriter(f"{os.getcwd() + folder_name}{map_name}.mp4", -1, 100.0, (320, 320))
    save_every = 100
    start = time.time()

    map_x, map_y = 32, 32
    state_novelty = np.zeros((map_x, map_y), dtype=np.uint32)

    if evaluate:
        maddpg_agents.load_checkpoint()

    done = True
    step = 0
    episode_step = 0
    obs = None
    episode_reward = []
    episode_im_reward = [[] for i in range(n_agents)]
    prev_agents_positions = []

    baseline_2v2 = [2, 4] * 10
    baseline_2v10 = [5] * 6 + [3] * 7

    while step < N_STEPS:
        if done:
            env.reset()
            if heat_map is None:
                heat_map = np.zeros((env.map_x * hm_size,
                                     env.map_y * hm_size))
            obs = env.get_obs()
            done = False
            episode_step = 0
            episode_reward = []
            episode_im_reward = [[] for i in range(n_agents)]

        # noise_rate = max(noise_rate_min, noise_rate - noise_decay_rate)
        noise_rate = max(noise_rate_min, sin_noise[step])
        agents_state_novelties = []
        heat_map *= 0.9998
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

        if not step % save_every:
            out.write(heatmap_image)

        cv2.imshow("Heatmap", heatmap_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        actions_probabilities = maddpg_agents.choose_action(obs, noise_rate)
        actions = [random.choices(np.arange(n_actions),
                                  weights=actions_probabilities[i] * env.get_avail_actions()[i])[0]
                   for i in range(n_agents)]
        avail_attack_actions = [[x for x in range(6, len(env.get_avail_agent_actions(i)))
                                 if env.get_avail_agent_actions(i)[x]] if env.agents[i].health > 0 else [0]
                                for i in env.agents.keys()]

        # baseline_2v2
        # actions = [baseline_2v2[episode_step] if not len(avail_attack_actions[i])
        #            else avail_attack_actions[i][0] for i in range(n_agents)]

        # baseline_2v10
        # actions = [baseline_2v10[episode_step] if episode_step < len(baseline_2v10)
        #            else avail_attack_actions[i][0] for i in range(n_agents)]

        agents_positions = [[env.agents[i].health > 0,
                             list(map(int, [env.agents[i].pos.x, env.agents[i].pos.y]))]
                            for i in env.agents.keys()]
        for alive, pos in agents_positions:
            if alive:
                state_novelty[pos[0], pos[1]] += 1

            agents_state_novelties.append([alive, state_novelty[pos[0], pos[1]]])

        sn_min, sn_max = state_novelty.min(), state_novelty.max()
        if not len(prev_agents_positions):
            prev_agents_positions = agents_positions.copy()

        intrinsic_rewards = []
        for agent_idx in range(n_agents):
            if prev_agents_positions[agent_idx][1] == agents_positions[agent_idx][1]:
                intrinsic_rewards.append(0)
            else:
                alive, agents_sn = agents_state_novelties[agent_idx]
                im_reward = (1 - (agents_sn - sn_min) / sn_max) ** 2 if alive else 0
                intrinsic_rewards.append(im_reward)

        # intrinsic_rewards = [(1 - (x[1] - sn_min) / sn_max) ** 2 * 2 if x[0] else 0 for x in agents_state_novelties]
        # intrinsic_rewards = [0 for _ in range(n_agents)]  # no IM

        reward, done, info = env.step(actions)
        obs_ = env.get_obs()
        prev_agents_positions = agents_positions.copy()

        episode_reward.append(reward)
        [episode_im_reward[i].append(intrinsic_rewards[i]) for i in range(n_agents)]

        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)

        if episode_step >= MAX_STEPS:
            done = True

        memory.store_transition(obs, state, actions_probabilities, reward, intrinsic_rewards, obs_, state_,
                                done)

        if step % 100 == 0 and not evaluate:
            maddpg_agents.learn(memory)

        obs = obs_

        writer.add_scalar('noise_rate', noise_rate, step)

        if done:
            sum_reward = sum(episode_reward)
            score_history.append(sum_reward)
            ep_len_history.append(episode_step)

            im_rewards_dict = {f"agent_{i}": sum(episode_im_reward[i]) for i in range(n_agents)}
            writer.add_scalar('Rewards/external_reward', sum_reward, step)
            writer.add_scalars('Rewards/IM_rewards', im_rewards_dict, step)
            writer.add_scalar('episode_length', episode_step, step)

            sn_img = np.zeros((env.map_x, env.map_y, 3), dtype=np.uint8)
            sn_img[:, :, 0] = (state_novelty / state_novelty.max() * 255).astype(np.uint8)
            sn_img = cv2.rotate(cv2.resize(sn_img, dsize=(env.map_x * hm_size, env.map_y * hm_size),
                                           interpolation=cv2.INTER_CUBIC), cv2.ROTATE_90_COUNTERCLOCKWISE)
            hm_img = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
            writer.add_image('Heatmaps/state_novelty_grid', sn_img, step, dataformats='HWC')
            writer.add_image('Heatmaps/agents_trajectories', hm_img, step, dataformats='HWC')

            avg_score = np.mean(score_history[-20:])
            if not evaluate:
                if avg_score > best_score:
                    maddpg_agents.save_checkpoint()
                    best_score = avg_score

        if step % PRINT_INTERVAL == 0 and step > 0:
            avg_score = np.mean(score_history[-20:])
            avg_ep_len = np.mean(ep_len_history[-20:])
            print('step', step, 'noise_rate', round(noise_rate, 5),
                  'average score {:.2f}'.format(avg_score),
                  'average episode length {:.2f}'.format(avg_ep_len))

        episode_step += 1
        step += 1

    print("End:", (time.time() - start) / 60)
    out.release()
    cv2.destroyAllWindows()
