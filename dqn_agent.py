"""
Deep Q-Network Agent - ULTIMATE AI!
PyTorch ile neural network tabanlı öğrenme
Experience Replay + Target Network + Double DQN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network - Sinir Ağı Mimarisi"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        # Optimized 3-layer network - Hızlı ve etkili!
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_size)
        )

        # Weight initialization (Xavier)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Learning Agent - TARİHİN EN İYİSİ!"""

    def __init__(self, config):
        """
        Args:
            config: Yapılandırma
        """
        self.config = config

        # Device (GPU varsa kullan!)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using: {self.device}")

        # State boyutu (sensör sayısı + 1 hız + 1 açı)
        sensor_count = config['car']['sensor_count']
        self.state_size = sensor_count + 2  # sensors + speed + angle
        self.action_size = 9  # 9 farklı aksiyon

        # Ana network (policy network)
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)

        # Target network (Double DQN için)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Training modunda değil

        # Optimizer (Adam - adaptive learning rate)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['learning'].get('learning_rate', 0.001)
        )

        # Loss function (Huber loss - outlier'lara dayanıklı)
        self.criterion = nn.SmoothL1Loss()

        # Experience Replay Buffer - Optimized (10 araba için)
        self.memory = deque(maxlen=10000)
        self.batch_size = 16  # Daha küçük batch = daha hızlı

        # Öğrenme parametreleri
        self.gamma = config['learning']['discount_factor']  # 0.99
        self.epsilon = config['learning']['exploration_rate']  # 1.0
        self.exploration_rate = self.epsilon  # Compatibility
        self.epsilon_decay = config['learning']['exploration_decay']  # 0.9995
        self.epsilon_min = config['learning']['min_exploration_rate']  # 0.1

        # Target network update frequency (daha az update = daha hızlı)
        self.target_update_freq = 2000
        self.steps_done = 0

        # İstatistikler
        self.total_states_explored = 0
        self.losses = []

    def get_state(self, car):
        """
        Arabanın durumunu neural network için hazırla

        Returns:
            torch.Tensor: State vektörü [sensörler + hız + açı]
        """
        # Sensör okumaları (7 değer, 0-1 arası normalize)
        sensors = car.get_sensor_readings()

        # Hız (0-1 arası normalize)
        speed = car.speed / car.max_speed

        # Açı (0-1 arası normalize, 0-360 -> 0-1)
        angle = (car.angle % 360) / 360.0

        # State vektörü oluştur
        state = sensors + [speed, angle]

        # Torch tensor'e çevir
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def choose_action(self, state, training=True):
        """
        Epsilon-greedy ile aksiyon seç

        Args:
            state: torch.Tensor state
            training: Eğitim modunda mı?

        Returns:
            int: Seçilen aksiyon (0-8)
        """
        # Exploration (rastgele)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation (en iyi aksiyon)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """
        Deneyimi hafızaya kaydet (Experience Replay)

        Args:
            state: Mevcut durum
            action: Yapılan aksiyon
            reward: Alınan ödül
            next_state: Yeni durum
            done: Epizod bitti mi?
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Experience Replay - Hafızadan öğren!
        Mini-batch gradient descent
        """
        if len(self.memory) < self.batch_size:
            return

        # Rastgele mini-batch seç
        batch = random.sample(self.memory, self.batch_size)

        # None state'leri filtrele (terminal states)
        states = []
        actions_list = []
        rewards_list = []
        next_states = []
        dones_list = []

        for s, a, r, ns, d in batch:
            if ns is not None:  # Sadece valid next_state olanları al
                states.append(s)
                actions_list.append(a)
                rewards_list.append(r)
                next_states.append(ns)
                dones_list.append(float(d))

        # Eğer valid sample yoksa çık
        if len(states) == 0:
            return

        states = torch.cat(states)
        actions = torch.LongTensor(actions_list).to(self.device)
        rewards = torch.FloatTensor(rewards_list).to(self.device)
        next_states = torch.cat(next_states)
        dones = torch.FloatTensor(dones_list).to(self.device)

        # Mevcut Q değerleri (policy network)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Hedef Q değerleri (target network) - Double DQN
        with torch.no_grad():
            # Policy network ile en iyi aksiyonu seç
            next_actions = self.policy_net(next_states).argmax(1)
            # Target network ile Q değerini hesapla
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss hesapla (Huber loss)
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (stability için)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # İstatistik
        self.losses.append(loss.item())
        self.total_states_explored += 1

        return loss.item()

    def update_target_network(self):
        """Target network'ü policy network ile güncelle"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_exploration(self):
        """Exploration rate'i azalt"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.exploration_rate = self.epsilon  # Sync

    def get_q_table(self):
        """Neural network ağırlıklarını döndür (kaydetmek için)"""
        return {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps_done
        }

    def set_q_table(self, state_dict):
        """Neural network ağırlıklarını yükle"""
        if 'policy_net' in state_dict:
            self.policy_net.load_state_dict(state_dict['policy_net'])
            self.target_net.load_state_dict(state_dict['target_net'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.epsilon = state_dict.get('epsilon', self.epsilon)
            self.steps_done = state_dict.get('steps', 0)
            print("Neural network yuklendi!")

    def reset_q_table(self):
        """Network'ü sıfırla"""
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory.clear()
        self.epsilon = 1.0
        self.steps_done = 0
        print("Neural network sifirlandi")

    def get_stats(self):
        """İstatistikleri döndür"""
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
        return {
            'exploration_rate': self.epsilon,
            'states_explored': self.total_states_explored,
            'memory_size': len(self.memory),
            'avg_loss': avg_loss,
            'device': str(self.device)
        }
