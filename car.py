"""
Araba sınıfı: Fizik, sensörler, çizim
"""
import pygame
import numpy as np
import math


class Car:
    """Simülasyonda kullanılan araba"""
    
    def __init__(self, x, y, angle, config, track_loader):
        """
        Args:
            x, y: Başlangıç konumu
            angle: Başlangıç açısı (derece)
            config: Yapılandırma
            track_loader: Pist yükleyici
        """
        self.x = x
        self.y = y
        self.angle = angle  # Derece cinsinden
        self.config = config
        self.track_loader = track_loader
        
        # Araba özellikleri
        self.width = config['car']['car_width']
        self.height = config['car']['car_height']
        self.speed = 0
        self.max_speed = config['car']['speed']
        self.min_speed = 0.5  # Minimum hız
        self.turn_angle = config['car']['turn_angle']

        # F1 Fizik Modeli - GERÇEKÇI!
        self.acceleration = 0.25  # F1 hızlı ivmelenir
        self.deceleration = 0.35  # Güçlü frenler
        self.tire_grip = 0.98  # Lastik tutuşu (0-1)
        self.downforce = 0.015  # Aerodinamik basınç
        self.drag_coefficient = 0.003  # Hava direnci
        self.lateral_friction = 0.92  # Yan kayma direnci
        
        # Sensör ayarları
        self.sensor_count = config['car']['sensor_count']
        self.sensor_length = config['car']['sensor_length']
        self.sensor_readings = [1.0] * self.sensor_count
        
        # Durum bilgileri
        self.alive = True
        self.distance_traveled = 0
        self.last_position = (x, y)

        # Renk
        self.color = (0, 255, 0)  # Yeşil

        # Performans optimizasyonu
        self.sensor_update_counter = 0
        self.sensor_update_interval = 4  # Her 4 frame'de bir sensör güncelle (10 araba için)
    
    def reset(self, x, y, angle):
        """Arabayı sıfırla"""
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.alive = True
        self.distance_traveled = 0
        self.last_position = (x, y)
        self.sensor_readings = [1.0] * self.sensor_count
    
    def update(self, action):
        """
        Arabayı güncelle - Gelişmiş hız kontrolü

        Args:
            action: Yapılacak eylem
            0: Düz git (sabit hız)
            1: Sağa dön
            2: Sola dön
            3: Tam gaz (hızlan)
            4: Fren (yavaşla)
            5: Sağa dön + hızlan
            6: Sola dön + hızlan
            7: Sağa dön + fren
            8: Sola dön + fren
        """
        if not self.alive:
            return

        # Başlangıç hızı (eğer 0 ise minimum hızda başla)
        if self.speed == 0:
            self.speed = self.min_speed

        # Eylem uygula
        if action == 0:  # Düz git - sabit hız
            pass
        elif action == 1:  # Sağa dön
            self.angle += self.turn_angle
        elif action == 2:  # Sola dön
            self.angle -= self.turn_angle
        elif action == 3:  # Tam gaz - F1 ivmelenme!
            accel = self.acceleration * self.tire_grip
            self.speed = min(self.speed + accel, self.max_speed)
        elif action == 4:  # Fren - F1 frenleme!
            decel = self.deceleration * self.tire_grip
            self.speed = max(self.speed - decel, self.min_speed)
        elif action == 5:  # Sağa + gaz - Agresif viraj
            self.angle += self.turn_angle * self.lateral_friction
            accel = self.acceleration * self.tire_grip * 0.9  # Virajda biraz yavaş
            self.speed = min(self.speed + accel, self.max_speed)
        elif action == 6:  # Sola + gaz - Agresif viraj
            self.angle -= self.turn_angle * self.lateral_friction
            accel = self.acceleration * self.tire_grip * 0.9
            self.speed = min(self.speed + accel, self.max_speed)
        elif action == 7:  # Sağa + fren - Trail braking!
            self.angle += self.turn_angle * self.lateral_friction * 1.1  # Frenle daha keskin dönüş
            decel = self.deceleration * self.tire_grip
            self.speed = max(self.speed - decel, self.min_speed)
        elif action == 8:  # Sola + fren - Trail braking!
            self.angle -= self.turn_angle * self.lateral_friction * 1.1
            decel = self.deceleration * self.tire_grip
            self.speed = max(self.speed - decel, self.min_speed)
        
        # Açıyı 0-360 arasında tut
        self.angle = self.angle % 360

        # F1 FİZİĞİ - Hava direnci ve downforce!
        # Hava direnci (yüksek hızda yavaşlatır)
        drag_force = self.drag_coefficient * (self.speed ** 2)
        self.speed = max(self.min_speed, self.speed - drag_force)

        # Downforce (hızla artar, virajlarda grip sağlar)
        downforce_effect = 1.0 + (self.downforce * self.speed)
        effective_grip = min(1.0, self.tire_grip * downforce_effect)

        # Hareketi hesapla
        rad = math.radians(self.angle)
        dx = self.speed * math.cos(rad) * effective_grip
        dy = self.speed * math.sin(rad) * effective_grip
        
        # Konumu güncelle
        self.last_position = (self.x, self.y)
        self.x += dx
        self.y += dy
        
        # Mesafe hesapla
        self.distance_traveled += math.sqrt(dx**2 + dy**2)
        
        # Pistte mi kontrol et
        if not self.track_loader.is_on_track(self.x, self.y):
            self.alive = False

        # Sensörleri güncelle (performans için her N frame'de bir)
        self.sensor_update_counter += 1
        if self.sensor_update_counter >= self.sensor_update_interval:
            self.update_sensors()
            self.sensor_update_counter = 0
    
    def update_sensors(self):
        """Sensörleri güncelle (ışınlar pistle çarpışma tespiti)"""
        self.sensor_readings = []
        
        # Her sensör için
        angle_step = 180 / (self.sensor_count - 1)  # -90 ile +90 arası dağıt
        
        for i in range(self.sensor_count):
            # Sensör açısı
            sensor_angle = self.angle + (-90 + i * angle_step)
            sensor_rad = math.radians(sensor_angle)
            
            # Işını takip et
            distance = self.cast_ray(sensor_rad)
            
            # Normalize et (0-1 arası)
            normalized_distance = min(distance / self.sensor_length, 1.0)
            self.sensor_readings.append(normalized_distance)
    
    def cast_ray(self, angle_rad):
        """
        Belirli bir açıda ışın gönder, pistle çarpışana kadar mesafeyi ölç

        Returns:
            Çarpışma mesafesi (piksel)
        """
        step = 10  # Her adımda kaç piksel ilerle (5'ten 10'a çıkarıldı - 2x hızlı)
        max_distance = self.sensor_length

        for distance in range(0, int(max_distance), step):
            # Işının uç noktası
            x = self.x + distance * math.cos(angle_rad)
            y = self.y + distance * math.sin(angle_rad)

            # Pistte mi kontrol et
            if not self.track_loader.is_on_track(x, y):
                return distance

        return max_distance
    
    def get_sensor_readings(self):
        """Sensör okumalarını döndür"""
        return self.sensor_readings
    
    def draw(self, screen, show_sensors=True):
        """Arabayı ekrana çiz (F1 stili)"""
        if not self.alive:
            return

        # F1 arabası şeklini çiz
        self.draw_f1_car(screen)

        # Sensörleri çiz
        if show_sensors:
            self.draw_sensors(screen)
    
    def get_car_points(self):
        """Araba köşe noktalarını hesapla (dikdörtgen)"""
        rad = math.radians(self.angle)

        # Dikdörtgen köşeleri için yön vektörleri
        # İleri/geri yönü (araba boyunca)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Yan yönü (perpendicular)
        cos_p = math.cos(rad + math.pi/2)
        sin_p = math.sin(rad + math.pi/2)

        # Dikdörtgen merkezden köşelere yarım boy/genişlik
        half_height = self.height / 2
        half_width = self.width / 2

        # 4 köşe noktası (saat yönünde: ön-sağ, ön-sol, arka-sol, arka-sağ)
        front_right_x = self.x + half_height * cos_a - half_width * cos_p
        front_right_y = self.y + half_height * sin_a - half_width * sin_p

        front_left_x = self.x + half_height * cos_a + half_width * cos_p
        front_left_y = self.y + half_height * sin_a + half_width * sin_p

        back_left_x = self.x - half_height * cos_a + half_width * cos_p
        back_left_y = self.y - half_height * sin_a + half_width * sin_p

        back_right_x = self.x - half_height * cos_a - half_width * cos_p
        back_right_y = self.y - half_height * sin_a - half_width * sin_p

        return [(front_right_x, front_right_y), (front_left_x, front_left_y),
                (back_left_x, back_left_y), (back_right_x, back_right_y)]

    def draw_f1_car(self, screen):
        """F1 arabası görünümünü çiz"""
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        cos_p = math.cos(rad + math.pi/2)
        sin_p = math.sin(rad + math.pi/2)

        # Ana gövde (monokok) - merkezdeki ana yapı
        body_length = self.height * 0.7
        body_width = self.width * 0.6
        body_points = [
            (self.x + body_length/2 * cos_a - body_width/2 * cos_p,
             self.y + body_length/2 * sin_a - body_width/2 * sin_p),
            (self.x + body_length/2 * cos_a + body_width/2 * cos_p,
             self.y + body_length/2 * sin_a + body_width/2 * sin_p),
            (self.x - body_length/2 * cos_a + body_width/2 * cos_p,
             self.y - body_length/2 * sin_a + body_width/2 * sin_p),
            (self.x - body_length/2 * cos_a - body_width/2 * cos_p,
             self.y - body_length/2 * sin_a - body_width/2 * sin_p)
        ]
        pygame.draw.polygon(screen, self.color, body_points)

        # Ön kanat (front wing)
        wing_offset = self.height * 0.45
        wing_width = self.width * 1.3
        wing_length = self.height * 0.08
        front_wing_points = [
            (self.x + wing_offset * cos_a - wing_width/2 * cos_p,
             self.y + wing_offset * sin_a - wing_width/2 * sin_p),
            (self.x + wing_offset * cos_a + wing_width/2 * cos_p,
             self.y + wing_offset * sin_a + wing_width/2 * sin_p),
            (self.x + (wing_offset - wing_length) * cos_a + wing_width/2 * cos_p,
             self.y + (wing_offset - wing_length) * sin_a + wing_width/2 * sin_p),
            (self.x + (wing_offset - wing_length) * cos_a - wing_width/2 * cos_p,
             self.y + (wing_offset - wing_length) * sin_a - wing_width/2 * sin_p)
        ]
        # Ön kanat daha koyu renk
        darker_color = tuple(max(0, c - 40) for c in self.color)
        pygame.draw.polygon(screen, darker_color, front_wing_points)

        # Arka kanat (rear wing)
        rear_offset = -self.height * 0.4
        rear_width = self.width * 1.2
        rear_length = self.height * 0.1
        rear_wing_points = [
            (self.x + rear_offset * cos_a - rear_width/2 * cos_p,
             self.y + rear_offset * sin_a - rear_width/2 * sin_p),
            (self.x + rear_offset * cos_a + rear_width/2 * cos_p,
             self.y + rear_offset * sin_a + rear_width/2 * sin_p),
            (self.x + (rear_offset + rear_length) * cos_a + rear_width/2 * cos_p,
             self.y + (rear_offset + rear_length) * sin_a + rear_width/2 * sin_p),
            (self.x + (rear_offset + rear_length) * cos_a - rear_width/2 * cos_p,
             self.y + (rear_offset + rear_length) * sin_a - rear_width/2 * sin_p)
        ]
        pygame.draw.polygon(screen, darker_color, rear_wing_points)

        # Kokpit (sürücü bölümü) - daha açık renk
        cockpit_offset = self.height * 0.1
        cockpit_length = self.height * 0.2
        cockpit_width = self.width * 0.4
        lighter_color = tuple(min(255, c + 60) for c in self.color)
        cockpit_points = [
            (self.x + cockpit_offset * cos_a - cockpit_width/2 * cos_p,
             self.y + cockpit_offset * sin_a - cockpit_width/2 * sin_p),
            (self.x + cockpit_offset * cos_a + cockpit_width/2 * cos_p,
             self.y + cockpit_offset * sin_a + cockpit_width/2 * sin_p),
            (self.x + (cockpit_offset - cockpit_length) * cos_a + cockpit_width/2 * cos_p,
             self.y + (cockpit_offset - cockpit_length) * sin_a + cockpit_width/2 * sin_p),
            (self.x + (cockpit_offset - cockpit_length) * cos_a - cockpit_width/2 * cos_p,
             self.y + (cockpit_offset - cockpit_length) * sin_a - cockpit_width/2 * sin_p)
        ]
        pygame.draw.polygon(screen, lighter_color, cockpit_points)

    def draw_sensors(self, screen):
        """Sensör ışınlarını çiz"""
        angle_step = 180 / (self.sensor_count - 1)
        
        for i, reading in enumerate(self.sensor_readings):
            # Sensör açısı
            sensor_angle = self.angle + (-90 + i * angle_step)
            sensor_rad = math.radians(sensor_angle)
            
            # Sensörün ulaştığı nokta
            distance = reading * self.sensor_length
            end_x = self.x + distance * math.cos(sensor_rad)
            end_y = self.y + distance * math.sin(sensor_rad)
            
            # Çizgi çiz (kırmızı = yakın, yeşil = uzak)
            color_intensity = int(255 * (1 - reading))
            color = (color_intensity, 255 - color_intensity, 0)
            
            pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 1)
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)