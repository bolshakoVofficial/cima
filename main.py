from smac.env import StarCraft2Env
from omegaconf import OmegaConf
from utils.episode_buffer import EpisodeBuffer
import torch
from agents.maddpg_agent import MADDPGAgent
from critics.maddpg_critic import MADDPGCritic

config = OmegaConf.load("configs/main_config.yaml")
env_args = OmegaConf.load("configs/sc2_config.yaml").env_args


def main():
    env = StarCraft2Env(**env_args)

    env_info = env.get_env_info()
    state_shape = env_info["state_shape"]
    obs_shape = env_info["obs_shape"]
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    episode_limit = env_info["episode_limit"]

    episode_buffer = EpisodeBuffer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actors = [MADDPGAgent(obs_shape, n_actions).to(device) for _ in range(config.n_agents)]
    target_actors = [MADDPGAgent(obs_shape, n_actions).to(device).load_state_dict(actors[i].state_dict())
                     for i in range(config.n_agents)]
    critic = MADDPGCritic(obs_shape * n_agents, n_agents).to(device)
    target_critic = MADDPGCritic(obs_shape * n_agents, n_agents).to(device).load_state_dict(critic.state_dict())

    print()


main()
