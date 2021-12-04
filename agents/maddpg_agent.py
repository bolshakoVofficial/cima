import torch.nn as nn


class MADDPGAgent(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(MADDPGAgent, self).__init__()
        # На вход нейронная сеть получает состояние среды для отдельного агента
        # На выходе нейронная сеть возвращает стратегию действий
        self.MADDPG_Actor = nn.Sequential(
            # Первый линейный слой обрабатывает входные данные состояния среды
            nn.Linear(obs_size, 60),
            nn.ReLU(),
            # Второй линейный слой обрабатывает внутренние данные
            nn.Linear(60, 60),
            nn.ReLU(),
            # Третий линейный слой обрабатывает внутренние данные
            nn.Linear(60, 60),
            nn.ReLU(),
            # Четвертый линейный слой обрабатывает данные для стратегии действий
            nn.Linear(60, n_actions)
        )
        # Финальный выход нерйонной сети обрабатывается функцией Tanh()
        self.tanh_layer = nn.Tanh()

    # Вначале данные x обрабатываются полносвязной сетью с функцией ReLU
    # На выходе происходит обработка функцией Tanh()
    def forward(self, x):
        # Обработка полносвязными линейными слоями
        network_out = self.MADDPG_Actor(x)
        # Обработка функцией Tanh()
        tanh_layer_out = self.tanh_layer(network_out)
        # Выход нейронной сети
        return tanh_layer_out
