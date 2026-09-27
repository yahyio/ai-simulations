"""
Yardımcı fonksiyonlar: Kaydetme, yükleme, log yazma
"""
import numpy as np
import json
import csv
import os
from datetime import datetime


def ensure_directories():
    """Gerekli klasörleri oluştur"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)


def save_q_table(q_table, filename='data/q_table.npy'):
    """Q-tablosunu kaydet"""
    np.save(filename, q_table)
    print(f"Q-table kaydedildi: {filename}")


def load_q_table(filename='data/q_table.npy'):
    """Q-tablosunu yükle"""
    if os.path.exists(filename):
        q_table = np.load(filename, allow_pickle=True).item()
        print(f"Q-table yuklendi: {filename}")
        return q_table
    else:
        print(f"Q-table bulunamadi, yeni tablo olusturuluyor")
        return {}


def save_stats(stats, filename='data/stats.json'):
    """İstatistikleri kaydet"""
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Istatistikler kaydedildi: {filename}")


def load_stats(filename='data/stats.json'):
    """İstatistikleri yükle"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            stats = json.load(f)
        print(f"Istatistikler yuklendi: {filename}")
        return stats
    else:
        print(f"Istatistik dosyasi bulunamadi, yeni baslaniyor")
        return {
            'total_episodes': 0,
            'best_lap_time': float('inf'),
            'total_crashes': 0,
            'completed_laps': 0
        }


def log_episode(episode, lap_time, reward_sum, crashes, filename='logs/times.csv'):
    """Her epizod sonrası CSV'ye kayıt yap"""
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # İlk kayıt ise başlık yaz
        if not file_exists:
            writer.writerow(['timestamp', 'episode', 'lap_time', 'reward_sum', 'crashes'])
        
        # Veriyi kaydet
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, episode, lap_time, reward_sum, crashes])


def get_recent_lap_times(count=10, filename='logs/times.csv'):
    """Son N turun ortalama süresini al"""
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if len(rows) == 0:
                return None
            
            # Son N turu al
            recent = rows[-count:]
            
            # lap_time sütunu var mı kontrol et
            lap_times = []
            for row in recent:
                try:
                    lap_time = float(row.get('lap_time', 0))
                    if lap_time > 0:
                        lap_times.append(lap_time)
                except (ValueError, KeyError):
                    continue
            
            if len(lap_times) == 0:
                return None
            
            return sum(lap_times) / len(lap_times)
    except Exception as e:
        print(f"get_recent_lap_times hatasi: {e}")
        return None

def discretize_sensor_readings(readings, bins=7):
    """
    Sensör okumalarını ayrıklaştır (Q-table için) - Daha hassas
    Sürekli değerleri kategorik hale getirir
    """
    discretized = []
    for reading in readings:
        # 0-1 arası normalize et
        normalized = min(reading, 1.0)
        # Bin'lere ayır (daha hassas)
        bin_index = int(normalized * bins)
        bin_index = min(bin_index, bins - 1)  # Maksimum bin'i aşma
        discretized.append(bin_index)

    return tuple(discretized)


def discretize_speed(speed, max_speed, bins=5):
    """Hızı ayrıklaştır - Geliştirilmiş"""
    normalized = min(speed / max_speed, 1.0)
    bin_index = int(normalized * bins)
    return min(bin_index, bins - 1)


def discretize_angle(angle, bins=16):
    """Açıyı ayrıklaştır (0-360 derece -> bins kategoriye) - Daha hassas yön algısı"""
    angle = angle % 360
    bin_index = int(angle / (360 / bins))
    return bin_index