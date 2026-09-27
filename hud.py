"""
HUD (Heads-Up Display): Ekran bilgileri gösterimi
İstatistikler, hız, tur süresi vb.
"""
import pygame
import time


class HUD:
    """Ekran bilgi paneli"""
    
    def __init__(self, config):
        """
        Args:
            config: Yapılandırma
        """
        self.config = config
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        
        # Renkler
        self.bg_color = (0, 0, 0, 180)  # Yarı saydam siyah
        self.text_color = (255, 255, 255)  # Beyaz
        self.highlight_color = (0, 255, 0)  # Yeşil
        self.warning_color = (255, 0, 0)  # Kırmızı
        
    def draw(self, screen, stats):
        """
        HUD'u ekrana çiz
        
        Args:
            screen: Pygame ekranı
            stats: İstatistik sözlüğü
        """
        if not self.config['display']['show_hud']:
            return
        
        # Panel boyutları
        panel_width = 350
        panel_height = 400
        panel_x = screen.get_width() - panel_width - 20
        panel_y = 20
        
        # Yarı saydam panel çiz
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, self.bg_color, (0, 0, panel_width, panel_height), border_radius=10)
        screen.blit(panel_surface, (panel_x, panel_y))
        
        # Başlık
        title = self.font_large.render("AI Racing", True, self.highlight_color)
        screen.blit(title, (panel_x + 20, panel_y + 15))
        
        # Çizgi
        pygame.draw.line(screen, self.text_color, 
                        (panel_x + 20, panel_y + 55), 
                        (panel_x + panel_width - 20, panel_y + 55), 2)
        
        # İstatistikler
        y_offset = panel_y + 70
        line_height = 35
        
        info_lines = [
            ("Episode:", f"{stats.get('episode', 0)}", self.text_color),
            ("Steps:", f"{stats.get('steps', 0)}", self.text_color),
            ("Speed:", f"{stats.get('speed', 0):.1f}", self.highlight_color),
            ("Distance:", f"{stats.get('distance', 0):.1f}", self.text_color),
            ("", "", self.text_color),  # Boşluk
            ("Lap Time:", f"{stats.get('current_lap_time', 0):.2f}s", self.highlight_color),
            ("Best Lap:", f"{stats.get('best_lap_time', 0):.2f}s", self.highlight_color),
            ("Avg Lap:", f"{stats.get('avg_lap_time', 0):.2f}s", self.text_color),
            ("", "", self.text_color),  # Boşluk
            ("Exploration:", f"{stats.get('exploration_rate', 0):.3f}", self.warning_color),
            ("States:", f"{stats.get('states_explored', 0)}", self.text_color),
            ("Completed:", f"{stats.get('completed_laps', 0)}", self.text_color),
        ]
        
        for label, value, color in info_lines:
            if label:  # Boş satırları atla
                # Etiket
                label_text = self.font_small.render(label, True, self.text_color)
                screen.blit(label_text, (panel_x + 20, y_offset))
                
                # Değer
                value_text = self.font_small.render(value, True, color)
                value_rect = value_text.get_rect()
                value_rect.right = panel_x + panel_width - 20
                value_rect.top = y_offset
                screen.blit(value_text, value_rect)
            
            y_offset += line_height
        
        # Durum göstergesi
        status = stats.get('status', 'Training')
        status_color = self.highlight_color if status == 'Training' else self.warning_color
        status_text = self.font_medium.render(f"[{status}]", True, status_color)
        status_rect = status_text.get_rect()
        status_rect.centerx = panel_x + panel_width // 2
        status_rect.top = panel_y + panel_height - 50
        screen.blit(status_text, status_rect)
    
    def draw_controls_help(self, screen):
        """Kontrol yardımını göster"""
        help_x = 20
        help_y = screen.get_height() - 120
        
        # Yarı saydam panel
        panel_width = 300
        panel_height = 100
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, self.bg_color, (0, 0, panel_width, panel_height), border_radius=10)
        screen.blit(panel_surface, (help_x, help_y))
        
        # Kontroller
        controls = [
            "SPACE - Pause/Resume",
            "R - Reset Learning",
            "S - Save Q-Table",
            "ESC - Menu"
        ]
        
        y_offset = help_y + 10
        for control in controls:
            text = self.font_small.render(control, True, self.text_color)
            screen.blit(text, (help_x + 15, y_offset))
            y_offset += 22
    
    def draw_message(self, screen, message, color=None):
        """Ekranın ortasına büyük mesaj göster"""
        if color is None:
            color = self.highlight_color
        
        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        
        # Arka plan
        bg_rect = text_rect.inflate(40, 20)
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, self.bg_color, (0, 0, bg_rect.width, bg_rect.height), border_radius=10)
        screen.blit(bg_surface, bg_rect)
        
        # Metin
        screen.blit(text, text_rect)