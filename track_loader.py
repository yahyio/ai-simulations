"""
Pist görselini yükleme ve analiz etme modülü
PNG dosyasından siyah (pist), beyaz (pist dışı) ve kırmızı (başlangıç) alanları tespit eder
"""
import pygame
import numpy as np


class TrackLoader:
    """Pist görselini yükler ve analiz eder"""
    
    def __init__(self, track_file, config):
        """
        Args:
            track_file: Pist görsel dosyası (PNG)
            config: Yapılandırma sözlüğü
        """
        self.track_file = track_file
        self.config = config
        
        # Renk tanımları
        self.track_color = tuple(config['track']['track_color'])
        self.off_track_color = tuple(config['track']['off_track_color'])
        self.start_line_color = tuple(config['track']['start_line_color'])
        
        # Görseli yükle
        self.image = None
        self.pixel_array = None
        self.width = 0
        self.height = 0
        self.start_position = None
        self.start_angle = 0
        
        self.load_track()
    
    def load_track(self):
        """Pist görselini yükle ve analiz et"""
        try:
            self.image = pygame.image.load(self.track_file)
            self.width = self.image.get_width()
            self.height = self.image.get_height()
            
            # Piksel dizisine çevir (hızlı erişim için)
            self.pixel_array = pygame.surfarray.array3d(self.image)
            
            # Başlangıç noktasını bul
            self.find_start_position()

            print(f"Pist yuklendi: {self.track_file}")
            print(f"   Boyut: {self.width}x{self.height}")
            print(f"   Baslangic: {self.start_position}, Aci: {self.start_angle}")

        except pygame.error as e:
            print(f"Pist yuklenemedi: {e}")
            raise
    
    def find_start_position(self):
        """Kırmızı başlangıç çizgisini bul"""
        red_pixels = []

        for x in range(self.width):
            for y in range(self.height):
                color = self.get_pixel_color(x, y)

                # Kırmızı piksel bulduk mu?
                if self.is_start_line(color):
                    red_pixels.append((x, y))

        if red_pixels:
            # Kırmızı piksellerin ortasını al
            avg_x = sum(p[0] for p in red_pixels) // len(red_pixels)
            avg_y = sum(p[1] for p in red_pixels) // len(red_pixels)
            self.start_position = (avg_x, avg_y)

            # Başlangıç açısını hesapla
            self.start_angle = self.calculate_start_angle(avg_x, avg_y)
            print(f"Kirmizi baslangic cizgisi bulundu: {len(red_pixels)} piksel")
            return

        # Kırmızı çizgi bulunamadıysa, pistin ilk siyah pikselini bul
        print("Kirmizi baslangic cizgisi bulunamadi, siyah piksel araniyor...")
        for x in range(self.width):
            for y in range(self.height):
                color = self.get_pixel_color(x, y)
                if self.is_track_color(color):
                    self.start_position = (x, y)
                    self.start_angle = 0
                    print(f"Siyah piksel kullanildi: ({x}, {y})")
                    return

        # Hiçbir şey bulunamadıysa ortayı kullan
        print("Hicbir uygun nokta bulunamadi, merkez kullaniliyor")
        self.start_position = (self.width // 2, self.height // 2)
        self.start_angle = 0
    
    def calculate_start_angle(self, x, y):
        """Başlangıç noktasındaki pist yönünü hesapla"""
        # Basit yaklaşım: yakındaki siyah piksellere bakarak yön belirle
        # Daha gelişmiş bir analiz de yapılabilir
        
        # Sağa doğru bakıyoruz varsayılan
        return 0
    
    def is_on_track(self, x, y):
        """Verilen koordinat pistte mi?"""
        if not self.is_valid_position(x, y):
            return False
        
        color = self.get_pixel_color(int(x), int(y))
        return self.is_track_color(color)
    
    def is_valid_position(self, x, y):
        """Koordinat görsel sınırları içinde mi?"""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_pixel_color(self, x, y):
        """Belirli bir pikselin rengini al"""
        if not self.is_valid_position(x, y):
            return self.off_track_color
        
        # pygame.surfarray RGB yerine BGR döndürür, düzelt
        return tuple(self.pixel_array[int(x), int(y)])
    
    def is_track_color(self, color):
        """Renk pist rengi mi? (siyah veya kırmızı)"""
        # Siyah veya kırmızı = pist
        return (self.color_distance(color, self.track_color) < 50 or
                self.color_distance(color, self.start_line_color) < 50)
    
    def is_start_line(self, color):
        """Renk başlangıç çizgisi mi? (kırmızı)"""
        return self.color_distance(color, self.start_line_color) < 50
    
    def color_distance(self, color1, color2):
        try:
            # Tuple'a çevir ve güvenli hesapla
            c1 = tuple(int(x) for x in color1[:3])  # İlk 3 değeri al (RGB)
            c2 = tuple(int(x) for x in color2[:3])

            distance = sum((a - b) ** 2 for a, b in zip(c1, c2))
            return distance ** 0.5
        except (TypeError, ValueError, OverflowError):
            return 999  # Hata durumunda büyük değer döndür
    
    def get_surface(self):
        """Pist görselini Surface olarak döndür"""
        return self.image
    
    def draw(self, screen, offset=(0, 0)):
        """Pisti ekrana çiz"""
        screen.blit(self.image, offset)