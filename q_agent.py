"""
Q-Learning Ajanı: Karar verme ve öğrenme mekanizması
Tablo bazlı Q-learning (sinir ağı yok)
"""
import numpy as np
import random
from utils import discretize_sensor_readings, discretize_speed, discretize_angle


class QLearningAgent:
    """Q-Learning tabanlı öğrenen ajan"""
    
    def __init__(self, config):
        """
        Args:
            config: Yapılandırma sözlüğü
        """
        self.config = config
        
        # Öğrenme parametreleri
        self.learning_rate = config['learning']['learning_rate']
        self.discount_factor = config['learning']['discount_factor']
        self.exploration_rate = config['learning']['exploration_rate']
        self.exploration_decay = config['learning']['exploration_decay']
        self.min_exploration_rate = config['learning']['min_exploration_rate']
        
        # Q-table (state -> action -> Q-value)
        self.q_table = {}

        # Gelişmiş eylem uzayı (9 aksiyon)
        # 0: Düz, 1: Sağ, 2: Sol, 3: Gaz, 4: Fren,
        # 5: Sağ+Gaz, 6: Sol+Gaz, 7: Sağ+Fren, 8: Sol+Fren
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.num_actions = len(self.actions)
        
        # İstatistikler
        self.total_states_explored = 0
        
    def get_state(self, car):
        """
        Arabanın mevcut durumunu ayrıklaştırılmış state'e çevir
        
        Args:
            car: Araba nesnesi
            
        Returns:
            Hashlenebilir state tuple'ı
        """
        # Sensör okumalarını ayrıklaştır
        sensor_bins = discretize_sensor_readings(car.get_sensor_readings(), bins=5)
        
        # Hızı ayrıklaştır
        speed_bin = discretize_speed(car.speed, car.max_speed, bins=3)
        
        # Açıyı ayrıklaştır
        angle_bin = discretize_angle(car.angle, bins=8)
        
        # State tuple'ı oluştur
        state = (sensor_bins, speed_bin, angle_bin)
        
        return state
    
    def get_q_value(self, state, action):
        """
        Belirli bir state-action çifti için Q-value döndür
        
        Args:
            state: Durum
            action: Eylem
            
        Returns:
            Q-value (yoksa 0.0)
        """
        if state not in self.q_table:
            return 0.0
        return self.q_table[state].get(action, 0.0)
    
    def choose_action(self, state, training=True):
        """
        Epsilon-greedy politika ile eylem seç
        
        Args:
            state: Mevcut durum
            training: Eğitim modunda mı? (False ise sadece en iyi eylemi seç)
            
        Returns:
            Seçilen eylem
        """
        # Keşif (exploration) vs. Sömürü (exploitation)
        if training and random.random() < self.exploration_rate:
            # Rastgele eylem (keşif)
            return random.choice(self.actions)
        else:
            # En iyi eylemi seç (sömürü)
            return self.get_best_action(state)
    
    def get_best_action(self, state):
        """
        Belirli bir state için en yüksek Q-value'ya sahip eylemi döndür
        
        Args:
            state: Durum
            
        Returns:
            En iyi eylem
        """
        if state not in self.q_table or len(self.q_table[state]) == 0:
            # Bu state daha önce görülmemişse rastgele seç
            return random.choice(self.actions)
        
        # En yüksek Q-value'ya sahip eylemi bul
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        
        # Aynı Q-value'ya sahip birden fazla eylem varsa rastgele seç
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Q-learning güncellemesi: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Mevcut durum
            action: Yapılan eylem
            reward: Alınan ödül
            next_state: Yeni durum
        """
        # Mevcut Q-value
        current_q = self.get_q_value(state, action)
        
        # Gelecekteki maksimum Q-value
        if next_state is not None:
            max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        else:
            # Terminal state (epizod bitti)
            max_next_q = 0.0
        
        # Yeni Q-value hesapla
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Q-table'ı güncelle
        if state not in self.q_table:
            self.q_table[state] = {}
            self.total_states_explored += 1
        
        self.q_table[state][action] = new_q
    
    def decay_exploration(self):
        """Her epizod sonrası exploration rate'i azalt"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def reset_exploration(self):
        """Exploration rate'i başlangıç değerine döndür"""
        self.exploration_rate = self.config['learning']['exploration_rate']
    
    def get_q_table(self):
        """Q-table'ı döndür"""
        return self.q_table
    
    def set_q_table(self, q_table):
        """Q-table'ı yükle"""
        self.q_table = q_table
        self.total_states_explored = len(q_table)
    
    def reset_q_table(self):
        """Q-table'ı sıfırla"""
        self.q_table = {}
        self.total_states_explored = 0
        self.reset_exploration()
        print("Q-table sifirlandi")
    
    def get_stats(self):
        """İstatistikleri döndür"""
        return {
            'exploration_rate': self.exploration_rate,
            'states_explored': self.total_states_explored,
            'q_table_size': len(self.q_table)
        }