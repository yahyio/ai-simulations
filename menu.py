"""
Pygame Menü Sistemi
Başlat, Durdur, Ayarlar, Çıkış
"""
import pygame
import sys


class Menu:
    """Ana menü"""
    
    def __init__(self, config, screen):
        """
        Args:
            config: Yapılandırma
            screen: Pygame ekranı
        """
        self.config = config
        self.screen = screen
        self.font_title = pygame.font.Font(None, 72)
        self.font_button = pygame.font.Font(None, 42)
        self.font_small = pygame.font.Font(None, 28)
        
        # Renkler
        self.bg_color = (20, 20, 30)
        self.button_color = (50, 50, 80)
        self.button_hover_color = (70, 70, 120)
        self.text_color = (255, 255, 255)
        self.highlight_color = (0, 255, 150)
        
        # Butonlar
        self.buttons = []
        self.create_buttons()
        
        self.selected_button = None
    
    def create_buttons(self):
        """Menü butonlarını oluştur"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        button_width = 300
        button_height = 60
        button_spacing = 20
        start_y = screen_height // 2 - 100
        
        button_labels = [
            "Start Training",
            "Load & Continue",
            "Reset Learning",
            "Settings",
            "Exit"
        ]
        
        self.buttons = []
        for i, label in enumerate(button_labels):
            x = screen_width // 2 - button_width // 2
            y = start_y + i * (button_height + button_spacing)
            button = {
                'label': label,
                'rect': pygame.Rect(x, y, button_width, button_height),
                'action': label.lower().replace(' ', '_')
            }
            self.buttons.append(button)
    
    def draw(self):
        """Menüyü çiz"""
        # Arka plan
        self.screen.fill(self.bg_color)
        
        # Başlık
        title = self.font_title.render("AI Racing Simulator", True, self.highlight_color)
        title_rect = title.get_rect(center=(self.screen.get_width() // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Alt başlık
        subtitle = self.font_small.render("Q-Learning Based Self-Learning Car", True, self.text_color)
        subtitle_rect = subtitle.get_rect(center=(self.screen.get_width() // 2, 160))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Butonlar
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.buttons:
            # Fare üzerinde mi?
            is_hover = button['rect'].collidepoint(mouse_pos)
            color = self.button_hover_color if is_hover else self.button_color
            
            # Buton çiz
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=10)
            pygame.draw.rect(self.screen, self.highlight_color, button['rect'], 3, border_radius=10)
            
            # Buton metni
            text = self.font_button.render(button['label'], True, self.text_color)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
        
        # Alt bilgi
        info_text = "Created with Pygame & Q-Learning | Press ESC anytime to return"
        info = self.font_small.render(info_text, True, (150, 150, 150))
        info_rect = info.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 30))
        self.screen.blit(info, info_rect)
    
    def handle_click(self, pos):
        """Tıklama olayını işle"""
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                return button['action']
        return None
    
    def run(self):
        """Menüyü çalıştır ve kullanıcı seçimini döndür"""
        clock = pygame.time.Clock()
        
        while True:
            # Olayları işle
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'exit'
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    action = self.handle_click(event.pos)
                    if action:
                        return action
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'exit'
            
            # Çiz
            self.draw()
            pygame.display.flip()
            clock.tick(30)


class SettingsMenu:
    """Ayarlar menüsü"""
    
    def __init__(self, config, screen):
        self.config = config
        self.screen = screen
        self.font_title = pygame.font.Font(None, 54)
        self.font_text = pygame.font.Font(None, 32)
        
        self.bg_color = (20, 20, 30)
        self.text_color = (255, 255, 255)
        self.highlight_color = (0, 255, 150)
        
        # Ayarlanabilir parametreler
        self.settings = [
            {'name': 'Learning Rate', 'key': ['learning', 'learning_rate'], 'min': 0.01, 'max': 1.0, 'step': 0.05},
            {'name': 'Discount Factor', 'key': ['learning', 'discount_factor'], 'min': 0.5, 'max': 0.99, 'step': 0.05},
            {'name': 'Exploration Decay', 'key': ['learning', 'exploration_decay'], 'min': 0.90, 'max': 0.999, 'step': 0.005},
            {'name': 'Sim Speed Ratio', 'key': ['simulation', 'sim_speed_ratio'], 'min': 1, 'max': 100, 'step': 5},
            {'name': 'Car Speed', 'key': ['car', 'speed'], 'min': 1.0, 'max': 10.0, 'step': 0.5},
            {'name': 'Turn Angle', 'key': ['car', 'turn_angle'], 'min': 1.0, 'max': 15.0, 'step': 1.0},
        ]
        
        self.selected_index = 0
    
    def get_value(self, keys):
        """İç içe dictionary'den değer al"""
        value = self.config
        for key in keys:
            value = value[key]
        return value
    
    def set_value(self, keys, new_value):
        """İç içe dictionary'ye değer ata"""
        current = self.config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = new_value
    
    def draw(self):
        """Ayarlar menüsünü çiz"""
        self.screen.fill(self.bg_color)
        
        # Başlık
        title = self.font_title.render("Settings", True, self.highlight_color)
        title_rect = title.get_rect(center=(self.screen.get_width() // 2, 60))
        self.screen.blit(title, title_rect)
        
        # Ayarlar
        y_offset = 150
        line_height = 70
        
        for i, setting in enumerate(self.settings):
            color = self.highlight_color if i == self.selected_index else self.text_color
            
            # Ayar adı
            name_text = self.font_text.render(setting['name'], True, color)
            self.screen.blit(name_text, (100, y_offset))
            
            # Değer
            value = self.get_value(setting['key'])
            value_text = self.font_text.render(f"{value:.3f}", True, color)
            value_rect = value_text.get_rect()
            value_rect.right = self.screen.get_width() - 100
            value_rect.top = y_offset
            self.screen.blit(value_text, value_rect)
            
            y_offset += line_height
        
        # Kontroller
        controls = self.font_text.render("Use Arrow Keys to adjust | Enter to save | ESC to cancel", True, (150, 150, 150))
        controls_rect = controls.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 40))
        self.screen.blit(controls, controls_rect)
    
    def run(self):
        """Ayarlar menüsünü çalıştır"""
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False  # İptal
                    
                    elif event.key == pygame.K_RETURN:
                        return True  # Kaydet
                    
                    elif event.key == pygame.K_UP:
                        self.selected_index = (self.selected_index - 1) % len(self.settings)
                    
                    elif event.key == pygame.K_DOWN:
                        self.selected_index = (self.selected_index + 1) % len(self.settings)
                    
                    elif event.key == pygame.K_LEFT:
                        # Değeri azalt
                        setting = self.settings[self.selected_index]
                        current = self.get_value(setting['key'])
                        new_value = max(setting['min'], current - setting['step'])
                        self.set_value(setting['key'], new_value)
                    
                    elif event.key == pygame.K_RIGHT:
                        # Değeri artır
                        setting = self.settings[self.selected_index]
                        current = self.get_value(setting['key'])
                        new_value = min(setting['max'], current + setting['step'])
                        self.set_value(setting['key'], new_value)
            
            self.draw()
            pygame.display.flip()
            clock.tick(30)