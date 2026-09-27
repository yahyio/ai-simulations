"""
Ana Program: Menü, simülasyon döngüsü, Q-learning entegrasyonu
"""
import pygame
import yaml
import sys
import time

from track_loader import TrackLoader
from env import Environment
from q_agent import QLearningAgent
from menu import Menu, SettingsMenu
from hud import HUD
from utils import (
    ensure_directories,
    save_q_table,
    load_q_table,
    save_stats,
    load_stats,
    log_episode,
    get_recent_lap_times
)


class RacingSimulator:
    """Ana simülatör sınıfı"""
    
    def __init__(self):
        """Simülatörü başlat"""
        # Pygame başlat
        pygame.init()
        
        # Yapılandırmayı yükle
        self.config = self.load_config()
        
        # Ekran oluştur (GPU acceleration ile)
        screen_width = self.config['display']['width']
        screen_height = self.config['display']['height']
        # HWSURFACE: Hardware surface, DOUBLEBUF: Double buffering, OPENGL için hazır
        self.screen = pygame.display.set_mode(
            (screen_width, screen_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED
        )
        pygame.display.set_caption("AI Racing Simulator - Q-Learning [GPU Accelerated]")
        
        # Gerekli klasörleri oluştur
        ensure_directories()
        
        # Bileşenler
        self.track_loader = None
        self.env = None
        self.agent = None
        self.hud = HUD(self.config)
        self.menu = Menu(self.config, self.screen)
        
        # Durum
        self.running = True
        self.paused = False
        self.training = False
        
        # İstatistikler
        self.stats = load_stats()
        self.current_episode = self.stats['total_episodes']
        
        # FPS
        self.clock = pygame.time.Clock()
        self.fps = self.config['simulation']['fps']

        # Simülasyon hızı
        self.sim_speed = self.config['simulation']['sim_speed_ratio']
        self.last_update_time = time.time()

        # Render optimizasyonu
        self.render_counter = 0
        self.render_interval = 1  # Her frame render et (değiştirilebilir)
    
    def load_config(self):
        """Yapılandırma dosyasını yükle"""
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("Yapılandırma yüklendi: config.yaml")
            return config
        except FileNotFoundError:
            print("❌ config.yaml bulunamadı!")
            sys.exit(1)
    
    def save_config(self):
        """Yapılandırmayı kaydet"""
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print("Yapilandirma kaydedildi")
    
    def initialize_simulation(self, load_existing=False):
        """Simülasyonu başlat"""
        # Pisti yükle
        track_file = self.config['track']['file']
        self.track_loader = TrackLoader(track_file, self.config)
        
        # Ortamı oluştur
        self.env = Environment(self.config, self.track_loader)
        
        # Ajanı oluştur
        self.agent = QLearningAgent(self.config)
        
        # Mevcut Q-table'ı yükle
        if load_existing:
            q_table = load_q_table()
            if q_table:
                self.agent.set_q_table(q_table)
                print(f"{len(q_table)} durum yuklendi")
    
    def run_menu(self):
        """Ana menüyü çalıştır"""
        action = self.menu.run()
        
        if action == 'start_training':
            self.initialize_simulation(load_existing=False)
            self.training = True
            return True
        
        elif action == 'load_&_continue':
            self.initialize_simulation(load_existing=True)
            self.training = True
            return True
        
        elif action == 'reset_learning':
            self.reset_learning()
            return False
        
        elif action == 'settings':
            settings_menu = SettingsMenu(self.config, self.screen)
            if settings_menu.run():
                self.save_config()
                print("Ayarlar kaydedildi")
            return False
        
        elif action == 'exit':
            self.quit()
            return False
        
        return False
    
    def reset_learning(self):
        """Öğrenmeyi sıfırla"""
        import os
        
        # Q-table'ı sil
        if os.path.exists('data/q_table.npy'):
            os.remove('data/q_table.npy')
        
        # İstatistikleri sıfırla
        self.stats = {
            'total_episodes': 0,
            'best_lap_time': float('inf'),
            'total_crashes': 0,
            'completed_laps': 0
        }
        save_stats(self.stats)
        
        # Log dosyasını sil
        if os.path.exists('logs/times.csv'):
            os.remove('logs/times.csv')

        print("Tum ogrenme verileri sifirlandi!")
        
        # Mesaj göster
        self.screen.fill((20, 20, 30))
        self.hud.draw_message(self.screen, "Learning Reset!", (255, 255, 0))
        pygame.display.flip()
        time.sleep(2)
    
    def training_loop(self):
        """Ana eğitim döngüsü"""
        # Ortamı sıfırla
        self.env.reset()
        episode_start_time = time.time()
        
        while self.running and self.training:
            # Olayları işle
            self.handle_events()
            
            if self.paused:
                # Durakladığında sadece ekranı güncelle
                self.screen.fill((20, 20, 30))
                self.track_loader.draw(self.screen)
                for car in self.env.cars:
                    car.draw(self.screen, self.config['display']['show_sensors'])
                self.hud.draw_message(self.screen, "PAUSED", (255, 255, 0))
                self.hud.draw_controls_help(self.screen)
                pygame.display.flip()
                self.clock.tick(30)
                continue
            
            # Simülasyon hızına göre güncelleme
            current_time = time.time()
            time_delta = current_time - self.last_update_time
            steps_to_run = int(time_delta * self.sim_speed * 60)  # 60 FPS baz alınarak

            if steps_to_run > 0:
                self.last_update_time = current_time

                for _ in range(min(steps_to_run, 10)):  # Maksimum 10 adım/frame
                    # Tüm arabaları paralel olarak güncelle
                    all_done, best_info = self.env.step_all()

                    # Epizod bitti mi?
                    if all_done:
                        self.episode_complete(best_info)
                        # Tüm agentların exploration rate'ini azalt
                        for agent in self.env.agents:
                            agent.decay_exploration()
                        self.env.reset()
                        episode_start_time = time.time()
                        break
            
            # Ekranı güncelle
            self.render()
            
            # FPS sınırla
            self.clock.tick(self.fps)
    
    def episode_complete(self, info):
        """Epizod tamamlandığında çağrılır"""
        self.current_episode += 1
        self.stats['total_episodes'] = self.current_episode
        
        # Tur tamamlandı mı?
        if info['lap_completed']:
            self.stats['completed_laps'] += 1
            lap_time = info['lap_time']
            
            # En iyi tur
            if lap_time < self.stats['best_lap_time']:
                self.stats['best_lap_time'] = lap_time
                print(f"🏆 Yeni rekor! {lap_time:.2f}s")
        else:
            self.stats['total_crashes'] += 1
            lap_time = 0
        
        # Log'a kaydet
        log_episode(
            self.current_episode,
            lap_time,
            info.get('best_reward', self.env.episode_reward),
            0 if info['lap_completed'] else 1
        )

        # Her 10 epizodda bir kaydet (en iyi agent'ın Q-table'ını kaydet)
        if self.current_episode % 10 == 0:
            # En fazla state keşfeden agent'ı bul ve kaydet
            if self.env.agents:
                best_agent = max(self.env.agents, key=lambda a: a.total_states_explored)
                save_q_table(best_agent.get_q_table())
            else:
                save_q_table(self.agent.get_q_table())
            save_stats(self.stats)
            print(f"💾 Epizod {self.current_episode}: Otomatik kayıt")
        
        # İlerlemeyi göster
        if self.current_episode % 5 == 0:
            avg_time = get_recent_lap_times(count=10)
            if avg_time:
                avg_epsilon = sum(a.exploration_rate for a in self.env.agents) / len(self.env.agents) if self.env.agents else 0
                print(f"📈 Epizod {self.current_episode} | Son 10 tur ort: {avg_time:.2f}s | Avg Epsilon: {avg_epsilon:.3f}")
    
    def render(self):
        """Ekranı çiz"""
        # Arka plan
        self.screen.fill((40, 40, 50))

        # Pist
        self.track_loader.draw(self.screen)

        # Tüm arabalar
        for car in self.env.cars:
            car.draw(self.screen, self.config['display']['show_sensors'])

        # HUD - en iyi araba için bilgileri göster
        alive_cars = [car for car in self.env.cars if car.alive]
        best_car = max(self.env.cars, key=lambda c: c.distance_traveled) if self.env.cars else self.env.car

        # Ortalama exploration rate
        avg_exploration = sum(a.exploration_rate for a in self.env.agents) / len(self.env.agents) if self.env.agents else self.agent.exploration_rate

        current_time = time.time() - self.env.start_times[0] if self.env.start_times else 0
        avg_lap = get_recent_lap_times(count=10)

        hud_stats = {
            'episode': self.current_episode,
            'steps': max(self.env.episode_steps) if isinstance(self.env.episode_steps, list) else self.env.episode_steps,
            'speed': best_car.speed if best_car else 0,
            'distance': best_car.distance_traveled if best_car else 0,
            'current_lap_time': current_time,
            'best_lap_time': self.stats['best_lap_time'] if self.stats['best_lap_time'] != float('inf') else 0,
            'avg_lap_time': avg_lap if avg_lap else 0,
            'exploration_rate': avg_exploration,
            'states_explored': sum(a.total_states_explored for a in self.env.agents) if self.env.agents else self.agent.total_states_explored,
            'completed_laps': self.stats['completed_laps'],
            'status': f"{'Paused' if self.paused else 'Training'} ({len(alive_cars)}/{self.env.num_cars} alive)"
        }

        self.hud.draw(self.screen, hud_stats)
        self.hud.draw_controls_help(self.screen)
        
        # Ekranı güncelle
        pygame.display.flip()
    
    def handle_events(self):
        """Olayları işle"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Menüye dön
                    self.training = False
                    save_q_table(self.agent.get_q_table())
                    save_stats(self.stats)
                    print("💾 Çıkış: Veriler kaydedildi")
                
                elif event.key == pygame.K_SPACE:
                    # Duraklat/Devam
                    self.paused = not self.paused
                    print("⏸️  Duraklatıldı" if self.paused else "▶️  Devam ediliyor")
                
                elif event.key == pygame.K_r:
                    # Öğrenmeyi sıfırla
                    self.paused = True
                    self.reset_learning()
                    self.agent.reset_q_table()
                    self.paused = False
                
                elif event.key == pygame.K_s:
                    # Manuel kaydet
                    save_q_table(self.agent.get_q_table())
                    save_stats(self.stats)
                    print("💾 Manuel kayıt yapıldı")
    
    def quit(self):
        """Programdan çık"""
        if self.agent:
            save_q_table(self.agent.get_q_table())
        if self.stats:
            save_stats(self.stats)
        
        print("👋 Program kapatılıyor...")
        pygame.quit()
        sys.exit(0)
    
    def run(self):
        """Ana programı çalıştır"""
        while self.running:
            # Menüyü göster
            start_training = self.run_menu()
            
            # Eğitim başlat
            if start_training:
                self.last_update_time = time.time()
                self.training_loop()
        
        self.quit()


if __name__ == "__main__":
    print("=" * 60)
    print("AI RACING SIMULATOR - Q-LEARNING")
    print("=" * 60)
    print()

    simulator = RacingSimulator()
    simulator.run()
