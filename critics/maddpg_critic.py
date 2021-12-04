import torch
import torch.nn as nn


class MADDPGCritic(nn.Module):
    def __init__(self, full_obs_size, n_actions_agents):
        super(MADDPGCritic, self).__init__()
        # На вход нейронная сеть получает состояние среды,
        # включающее все локальные состояния среды от отдельных агентов
        # и все выполненные действия отдельных агентов
        # На выходе нейронная сеть возвращает корректирующее значение
        self.network = nn.Sequential(
            # Первый линейный слой обрабатывает входные данные
            nn.Linear(full_obs_size + n_actions_agents, 202),
            nn.ReLU(),
            # Второй линейный слой обрабатывает внутренние данные
            nn.Linear(202, 60),
            nn.ReLU(),
            # Третий линейный слой обрабатывает внутренние данные
            nn.Linear(60, 30),
            nn.ReLU(),
            # Четвертый линейный слой обрабатывает выходные данные
            nn.Linear(30, 1)
        )

    # Данные x последовательно обрабатываются полносвязной сетью с функцией ReLU
    def forward(self, state, action):
        # Объединяем данные состояний и действий для передачи в сеть
        x = torch.cat([state, action], dim=2)
        # Результаты обработки
        q_value = self.network(x)
        # Финальный выход нейронной сети
        return q_value
