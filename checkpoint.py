"""
Checkpoint sistemi - TrackMania tarzı
Arabalar pistin belirli noktalarından sırayla geçmek zorunda
"""
import pygame
import math


class CheckpointSystem:
    """Checkpoint yönetimi"""

    def __init__(self, track_image):
        """
        Args:
            track_image: Pist görüntüsü (pygame Surface)
        """
        self.track_image = track_image
        self.width = track_image.get_width()
        self.height = track_image.get_height()

        # Checkpoint'leri otomatik bul (yeşil pikseller)
        self.checkpoints = self.find_checkpoints()
        self.num_checkpoints = len(self.checkpoints)

        print(f"Checkpoint sistemi: {self.num_checkpoints} checkpoint bulundu")
        for i, cp in enumerate(self.checkpoints):
            print(f"  Checkpoint {i+1}: ({cp['x']}, {cp['y']}) - {cp['radius']} piksel")

    def find_checkpoints(self):
        """Pistteki yeşil pikselleri checkpoint olarak işaretle"""
        checkpoints = []
        pixel_array = pygame.surfarray.array3d(self.track_image)

        # Yeşil pikselleri bul (0, 255, 0)
        green_pixels = []
        for x in range(self.width):
            for y in range(self.height):
                r, g, b = pixel_array[x, y]
                # Yeşil mi? (g dominant, r ve b düşük)
                if g > 200 and r < 100 and b < 100:
                    green_pixels.append((x, y))

        if len(green_pixels) == 0:
            print("UYARI: Pistte checkpoint (yesil piksel) bulunamadi!")
            print("Otomatik checkpoint'ler olusturuluyor...")
            return self.create_auto_checkpoints()

        # Yeşil pikselleri gruplara ayır (yakın olanlar aynı checkpoint)
        clusters = self.cluster_pixels(green_pixels, max_distance=50)

        # Her cluster için merkez nokta ve yarıçap hesapla
        for i, cluster in enumerate(clusters):
            center_x = sum(p[0] for p in cluster) // len(cluster)
            center_y = sum(p[1] for p in cluster) // len(cluster)

            # Yarıçap: en uzak piksele olan mesafe
            max_dist = max(math.hypot(p[0] - center_x, p[1] - center_y) for p in cluster)
            radius = max(30, max_dist + 20)  # Minimum 30 piksel

            checkpoints.append({
                'id': i,
                'x': center_x,
                'y': center_y,
                'radius': radius
            })

        # Checkpoint'leri pist etrafında saat yönünde sırala
        checkpoints = self.sort_checkpoints_clockwise(checkpoints)

        return checkpoints

    def cluster_pixels(self, pixels, max_distance=50):
        """Yakın pikselleri gruplara ayır"""
        if not pixels:
            return []

        clusters = []
        remaining = list(pixels)

        while remaining:
            # Yeni cluster başlat
            current_cluster = [remaining.pop(0)]

            # Yakın pikselleri bul
            i = 0
            while i < len(remaining):
                px, py = remaining[i]
                # Bu piksel cluster'daki herhangi birine yakın mı?
                close_to_cluster = any(
                    math.hypot(px - cx, py - cy) < max_distance
                    for cx, cy in current_cluster
                )

                if close_to_cluster:
                    current_cluster.append(remaining.pop(i))
                else:
                    i += 1

            clusters.append(current_cluster)

        return clusters

    def sort_checkpoints_clockwise(self, checkpoints):
        """Checkpoint'leri pistin merkezi etrafında saat yönünde sırala"""
        if not checkpoints:
            return []

        # Pistin merkezi
        center_x = self.width // 2
        center_y = self.height // 2

        # Her checkpoint için açı hesapla
        for cp in checkpoints:
            angle = math.atan2(cp['y'] - center_y, cp['x'] - center_x)
            cp['angle'] = angle

        # Açıya göre sırala
        sorted_cps = sorted(checkpoints, key=lambda cp: cp['angle'])

        # ID'leri yeniden ata
        for i, cp in enumerate(sorted_cps):
            cp['id'] = i
            del cp['angle']

        return sorted_cps

    def create_auto_checkpoints(self):
        """Eğer yeşil piksel yoksa otomatik checkpoint oluştur"""
        checkpoints = []
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) // 3

        # 8 checkpoint oluştur (pistin etrafında)
        for i in range(8):
            angle = (i * 2 * math.pi) / 8
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))

            checkpoints.append({
                'id': i,
                'x': x,
                'y': y,
                'radius': 50
            })

        return checkpoints

    def check_collision(self, car_x, car_y, checkpoint_id):
        """Araba checkpoint'e değdi mi?"""
        if checkpoint_id >= self.num_checkpoints:
            return False

        cp = self.checkpoints[checkpoint_id]
        distance = math.hypot(car_x - cp['x'], car_y - cp['y'])

        return distance < cp['radius']

    def get_distance_to_checkpoint(self, car_x, car_y, checkpoint_id):
        """Araba ile checkpoint arası mesafe"""
        if checkpoint_id >= self.num_checkpoints:
            return float('inf')

        cp = self.checkpoints[checkpoint_id]
        return math.hypot(car_x - cp['x'], car_y - cp['y'])

    def draw(self, screen):
        """Checkpoint'leri ekrana çiz (debug için)"""
        for cp in self.checkpoints:
            # Checkpoint dairesi
            pygame.draw.circle(screen, (0, 255, 0), (cp['x'], cp['y']), cp['radius'], 2)
            # Numara
            font = pygame.font.Font(None, 24)
            text = font.render(str(cp['id'] + 1), True, (255, 255, 255))
            screen.blit(text, (cp['x'] - 8, cp['y'] - 8))
