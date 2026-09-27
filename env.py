"""
Simülasyon Ortamı (Environment)
Araba, pist ve reward sistemi - TrackMania tarzı
"""
import time
import math
from car import Car
from q_agent import QLearningAgent
from checkpoint import CheckpointSystem
try:
    from dqn_agent import DQNAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("DQN agent not available, using Q-Learning")


class Environment:
    """Simülasyon ortamı"""
    
    def __init__(self, config, track_loader):
        """
        Args:
            config: Yapılandırma
            track_loader: Pist yükleyici
        """
        self.config = config
        self.track_loader = track_loader

        # Checkpoint sistemi - TrackMania tarzı!
        self.checkpoint_system = CheckpointSystem(track_loader.image)

        # Araba sayısı
        self.num_cars = config['car'].get('num_cars', 1)

        # Arabalar ve agentlar oluştur
        start_x, start_y = track_loader.start_position
        start_angle = track_loader.start_angle

        self.cars = []
        self.agents = []
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                  (0, 255, 255), (255, 128, 0), (128, 0, 255), (255, 255, 255), (128, 128, 128)]

        # Agent type seçimi
        agent_type = config['learning'].get('agent_type', 'qlearning')
        use_dqn = (agent_type == 'dqn' and DQN_AVAILABLE)

        if use_dqn:
            print("DEEP Q-NETWORK (DQN) aktif - Neural Network kullaniliyor!")
        else:
            print("Q-Learning aktif - Q-table kullaniliyor")

        for i in range(self.num_cars):
            # Her araba için ayrı Car nesnesi
            car = Car(start_x, start_y, start_angle, config, track_loader)
            car.color = colors[i % len(colors)]
            car.car_id = i  # Araba ID'si ekle
            self.cars.append(car)

            # Her araba için ayrı Agent (DQN veya Q-Learning)
            if use_dqn:
                agent = DQNAgent(config)
            else:
                agent = QLearningAgent(config)
            self.agents.append(agent)

        # Eski arayüz uyumluluğu için
        self.car = self.cars[0] if self.cars else None

        # Epizod bilgileri (her araba için)
        self.episode_steps = [0] * self.num_cars
        self.episode_rewards = [0] * self.num_cars
        self.max_steps = config['simulation']['max_steps_per_episode']

        # Zaman (her araba için)
        self.start_times = [time.time()] * self.num_cars
        self.lap_times = [0] * self.num_cars

        # Tur tamamlama (her araba için) - ESKİ SİSTEM KALDIRILDI
        self.lap_completed = [False] * self.num_cars

        # CHECKPOINT SİSTEMİ - TrackMania tarzı!
        self.current_checkpoint = [0] * self.num_cars  # Her arabanın şu anki hedef checkpoint'i
        self.checkpoints_passed = [0] * self.num_cars  # Kaç checkpoint geçti
        self.last_checkpoint_time = [time.time()] * self.num_cars  # Son checkpoint'ten geçiş zamanı
        self.prev_distance_to_checkpoint = [float('inf')] * self.num_cars  # Checkpoint'e yaklaşıyor mu?
        self.wrong_way_counter = [0] * self.num_cars  # Ters yöne gitme sayacı

        # Eski uyumluluk
        self.episode_reward = 0
        
    def reset(self):
        """Ortamı sıfırla (yeni epizod)"""
        start_x, start_y = self.track_loader.start_position
        start_angle = self.track_loader.start_angle

        for car in self.cars:
            car.reset(start_x, start_y, start_angle)

        # Her araba için sıfırla
        current_time = time.time()
        self.episode_steps = [0] * self.num_cars
        self.episode_rewards = [0] * self.num_cars
        self.start_times = [current_time] * self.num_cars
        self.lap_times = [0] * self.num_cars
        self.lap_completed = [False] * self.num_cars

        # Checkpoint sistemi sıfırla
        self.current_checkpoint = [0] * self.num_cars
        self.checkpoints_passed = [0] * self.num_cars
        self.last_checkpoint_time = [current_time] * self.num_cars
        self.prev_distance_to_checkpoint = [float('inf')] * self.num_cars
        self.wrong_way_counter = [0] * self.num_cars

        # Eski uyumluluk
        self.episode_reward = 0

        return self.get_state()
    
    def step(self, action):
        """
        Bir adım simülasyon çalıştır
        
        Args:
            action: Yapılacak eylem
            
        Returns:
            next_state: Yeni durum
            reward: Ödül
            done: Epizod bitti mi?
            info: Ek bilgiler
        """
        # Arabayı güncelle
        self.car.update(action)
        self.episode_steps += 1
        
        # Ödülü hesapla
        reward = self.calculate_reward()
        self.episode_reward += reward
        
        # Epizod bitti mi?
        done = False
        
        # Pistten çıktı mı?
        if not self.car.alive:
            reward += self.config['rewards']['off_track']
            done = True
        
        # Maksimum adım sayısı aşıldı mı?
        if self.episode_steps >= self.max_steps:
            done = True
        
        # Tur tamamlandı mı?
        if self.lap_completed:
            reward += self.config['rewards']['lap_complete']
            self.lap_time = time.time() - self.start_time
            done = True
        
        # Yeni durum
        next_state = self.get_state() if not done else None
        
        # Ek bilgiler
        info = {
            'lap_completed': self.lap_completed,
            'lap_time': self.lap_time,
            'steps': self.episode_steps,
            'distance': self.car.distance_traveled
        }
        
        return next_state, reward, done, info
    
    def calculate_reward(self):
        """Ödülü hesapla"""
        reward = 0
        
        # Pistte kalmak için küçük ödül
        if self.car.alive:
            reward += self.config['rewards']['stay_on_track']
        
        # İlerleme ödülü (başlangıç noktasından uzaklaşma)
        distance_from_start = self.distance_to_start()
        
        if distance_from_start < self.min_distance_from_start:
            # Geriye gidiyor
            reward += self.config['rewards']['backwards']
        else:
            # İlerliyor
            reward += self.config['rewards']['forward_progress']
            self.min_distance_from_start = distance_from_start
        
        # Tur tamamlama kontrolü
        self.check_lap_completion()
        
        return reward
    
    def distance_to_start(self):
        """Başlangıç noktasına mesafe"""
        start_x, start_y = self.track_loader.start_position
        dx = self.car.x - start_x
        dy = self.car.y - start_y
        return (dx**2 + dy**2) ** 0.5
    
    def check_lap_completion(self):
        """Tur tamamlandı mı kontrol et"""
        # Başlangıç çizgisine yakınlık
        distance = self.distance_to_start()
        
        # İlk kez başlangıç çizgisinden uzaklaştı mı?
        if not self.crossed_start_line and distance > 50:
            self.crossed_start_line = True
        
        # Başlangıç çizgisinden uzaklaştıktan sonra tekrar yaklaştı mı?
        if self.crossed_start_line and distance < 30:
            # Tur tamamlandı!
            self.lap_completed = True
    
    def step_all(self):
        """
        Tüm arabaları paralel olarak bir adım ilerlet
        Her araba kendi agent'ı ile karar verir ve öğrenir

        Returns:
            all_done: Tüm arabalar bitirdiyse True
            best_info: En iyi performans bilgisi
        """
        all_done = True

        for i, (car, agent) in enumerate(zip(self.cars, self.agents)):
            # Araba zaten öldüyse veya turu tamamladıysa atla
            if not car.alive or self.lap_completed[i]:
                continue

            all_done = False

            # Mevcut durumu al
            state = agent.get_state(car)

            # Agent'tan eylem al
            action = agent.choose_action(state, training=True)

            # Arabayı güncelle
            car.update(action)
            self.episode_steps[i] += 1

            # Ödülü hesapla
            reward = self.calculate_reward_for_car(i)
            self.episode_rewards[i] += reward

            # Yeni durumu al
            next_state = agent.get_state(car) if car.alive else None

            # DQN mı Q-Learning mi?
            if hasattr(agent, 'remember'):  # DQN
                # Experience Replay buffer'a kaydet
                done = not car.alive or self.lap_completed[i]
                agent.remember(state, action, reward, next_state, done)

                # Her 8 adımda bir öğren (10 araba için optimize)
                agent.steps_done += 1
                if agent.steps_done % 8 == 0:
                    agent.replay()

                # Target network'ü periyodik güncelle
                if agent.steps_done % agent.target_update_freq == 0:
                    agent.update_target_network()

            else:  # Q-Learning
                # Q-value güncelle (eski yöntem)
                agent.update_q_value(state, action, reward, next_state)

            # Pistten çıktı mı?
            if not car.alive:
                reward += self.config['rewards']['off_track']
                self.episode_rewards[i] += self.config['rewards']['off_track']

            # Tur tamamlandı mı?
            if self.lap_completed[i]:
                reward += self.config['rewards']['lap_complete']
                self.episode_rewards[i] += self.config['rewards']['lap_complete']
                self.lap_times[i] = time.time() - self.start_times[i]

            # Maksimum adım aşıldı mı?
            if self.episode_steps[i] >= self.max_steps:
                car.alive = False

        # En iyi bilgileri topla
        best_info = self.get_best_car_info()

        # Eski uyumluluk için
        if self.cars:
            self.episode_reward = self.episode_rewards[0]

        return all_done, best_info

    def calculate_reward_for_car(self, car_idx):
        """
        TRACKMANIA TARZI REWARD - EN OPTIMAL ROTAYI BUL!
        Checkpoint'lerden sırayla geçmek zorunlu!
        """
        car = self.cars[car_idx]
        reward = 0

        if not car.alive:
            return reward

        # 1. BASE REWARD - Hayatta kalmak
        reward += 0.1

        # 2. CHECKPOINT SİSTEMİ - Ana ilerleme metrigi!
        current_cp = self.current_checkpoint[car_idx]
        distance_to_cp = self.checkpoint_system.get_distance_to_checkpoint(car.x, car.y, current_cp)

        # Checkpoint'e ulaştı mı?
        if self.checkpoint_system.check_collision(car.x, car.y, current_cp):
            # CHECKPOINT GECILDI! MUAZZAM ODUL!
            checkpoint_reward = 1000.0

            # Hızlı geçtiyse bonus!
            current_time = time.time()
            time_since_last_cp = current_time - self.last_checkpoint_time[car_idx]
            if time_since_last_cp < 3.0:  # 3 saniyeden hızlıysa
                checkpoint_reward *= (1.0 + (3.0 - time_since_last_cp) / 3.0)  # 2x bonus!

            reward += checkpoint_reward

            # Bir sonraki checkpoint'e geç
            self.checkpoints_passed[car_idx] += 1
            self.current_checkpoint[car_idx] = (current_cp + 1) % self.checkpoint_system.num_checkpoints
            self.last_checkpoint_time[car_idx] = current_time
            self.prev_distance_to_checkpoint[car_idx] = float('inf')

            # Tüm checkpoint'leri geçtiyse TUR TAMAMLANDI!
            if self.checkpoints_passed[car_idx] >= self.checkpoint_system.num_checkpoints:
                reward += 10000.0  # MEGA BONUS!
                self.lap_completed[car_idx] = True

        else:
            # Checkpoint'e yaklaşıyor mu?
            prev_dist = self.prev_distance_to_checkpoint[car_idx]
            if distance_to_cp < prev_dist:
                # Yaklaşıyor - İYİ!
                progress = prev_dist - distance_to_cp
                reward += progress * 0.5  # Her piksel için 0.5 puan

                # Hızlı yaklaşıyorsa daha çok ödül!
                speed_bonus = (car.speed / car.max_speed) * 2.0
                reward += speed_bonus
            else:
                # Uzaklaşıyor - KÖTÜ!
                reward -= 5.0
                self.wrong_way_counter[car_idx] += 1

                # Çok uzun süre ters giderse MEGA CEZA!
                if self.wrong_way_counter[car_idx] > 30:
                    reward -= 50.0

            self.prev_distance_to_checkpoint[car_idx] = distance_to_cp

            # Checkpoint'e çok uzaksa veya çok uzun süredir geçilmediyse CEZA
            time_since_last_cp = time.time() - self.last_checkpoint_time[car_idx]
            if time_since_last_cp > 10.0:  # 10 saniye geçti ama hala checkpoint geçilmedi
                reward -= 10.0

        # 3. HIZ BONUSU - Hızlı gitmek her zaman iyi!
        speed_ratio = car.speed / car.max_speed
        reward += speed_ratio * 1.0

        # 4. SMOOTH DRIVING BONUSU
        # Ani dönüşler kötü, smooth driving iyi
        # (Bu kısım gelecekte eklenebilir - şimdilik sadece hız ve checkpoint önemli)

        return reward

    def get_best_car_info(self):
        """En iyi performans gösteren araba bilgisini döndür"""
        best_distance = 0
        best_idx = 0
        best_lap_time = float('inf')
        any_completed = False

        for i, car in enumerate(self.cars):
            # Turu tamamlayan var mı?
            if self.lap_completed[i]:
                any_completed = True
                if self.lap_times[i] < best_lap_time:
                    best_lap_time = self.lap_times[i]
                    best_idx = i
            elif car.distance_traveled > best_distance:
                best_distance = car.distance_traveled
                best_idx = i

        return {
            'best_car_id': best_idx,
            'lap_completed': any_completed,
            'lap_time': best_lap_time if any_completed else 0,
            'best_distance': best_distance,
            'best_reward': self.episode_rewards[best_idx] if best_idx < len(self.episode_rewards) else 0
        }

    def get_state(self):
        """Mevcut durumu döndür"""
        return self.car

    def render(self, screen):
        """Ortamı ekrana çiz"""
        # Pisti çiz
        self.track_loader.draw(screen)

        # Tüm arabaları çiz
        show_sensors = self.config['display']['show_sensors']
        for car in self.cars:
            car.draw(screen, show_sensors=show_sensors)