# AI simülasyonları

Bu klasörde pekiştirmeli öğrenmeyle sıfırdan öğrenen iki simülasyon var:

| | Başlatma | Ne öğreniyor |
|---|---|---|
| **AI Racing** | `python main.py` | 16 araba pistte tur atmayı öğrenir; pistler rastgele üretilebilir |
| **AI Walker** | `python yurume.py` | Hümanoid karakter hiçbir şey bilmeden yürümeyi / koşmayı öğrenir |

---

# AI Walker — deneme yanılmayla yürüyen hümanoid

```bash
python yurume.py
```

Karakter (59 kg, 1.68 m; kafa, gövde, bel, iki kol, iki bacak — **11 motorlu eklem**) başlangıçta
hiçbir şey bilmez: eklemlerine rastgele tork uygular ve yere yığılır. Tek geri bildirimi bir sayıdır
(ödül): ileri gittikçe artar, düşünce deneme biter. 32 karakter aynı anda dener (ekranda yarı saydam
"hayaletler"), en öndeki net çizilir ve kamera onu izler.

- **Yürümeyi öğren**: hedef hız 1.5 m/s; hızlısı ödüllendirilmez, sakin ve dengeli bir yürüyüş çıkar.
- **Koşmayı öğren**: olabildiğince hızlı.

Ödül yalnızca "ne kadar iyi" olduğunu söyler, "nasıl" yapılacağını değil: ileri hız, ayakta kalma,
dik duruş (eğiklik cezası), uzun ve **sağ-sol sırayla** atılan adımlar (aynı ayakla sekme cezalı) ve
az enerji. Bu terimler olmadan ajan belden 45° geriye yatıp sürünmeyi ya da tek ayakla sekmeyi
"keşfediyordu".

Ölçülen gelişim (varsayılan ayarlar, pencere olmadan, tek çekirdekte ~4.500 adım/sn):

| Adım | Yürüme | Koşu |
|---|---|---|
| 0.1 M | ~1 sn içinde yere yığılıyor | aynı |
| 0.45 M | sendeliyor, ~2 sn | ~4 sn, 1.6 m/s |
| 1 M | 15 sn ayakta, 0.9 m/s | 3.8 m/s |
| 2 M | **hiç düşmüyor**, 1.3 m/s | 5.2 m/s |
| 3 M | dik, uzun adımlı yürüyüş, 1.45 m/s | **5.8 m/s** (~21 km/s), 16 sn'de 89 m |

Pencerede turbo (`V`) ile ilk dengeli adımlar birkaç dakikada, düzgün yürüyüş ~10–15 dakikada görülür. Tuşlar: `Boşluk` duraklat,
`+`/`-` hız, `V` turbo, `G` hayaletleri gizle, `S` kaydet, `Esc` menü.

Pencere olmadan (en hızlı):

```bash
python yurume.py --headless --task yurume --steps 2000000
```

Teknik: MuJoCo fizik motoru, düzlemsel (2B) hümanoid, PPO (gözlem normalizasyonu, ödül ölçekleme,
GAE, zaman sınırında değer tahmini). Modeller `data/walker/<görev>/`, günlükler `logs/walker_<görev>.csv`.

---

# AI Racing — kendi kendine süren arabalar

Bir pist görselinde, **pekiştirmeli öğrenme (Dueling Double DQN)** ile sürmeyi sıfırdan öğrenen arabalar.
16 araba aynı "beyni" paylaşır; hepsi aynı anda deneyim toplar. Varsayılan pistte ilk tur genellikle
**5. bölüm civarında** gelir, birkaç bölüm sonra arabaların neredeyse tamamı turu bitirir.

## Kurulum

```bash
pip install -r requirements.txt
```

Python 3.10+ gerekir.

## Çalıştırma

```bash
python main.py
```

Menüden **Yeni eğitim**'i seçin. Eğitim sırasında:

| Tuş | İşlev |
|---|---|
| `Boşluk` | duraklat / devam |
| `+` / `-` | simülasyon hızı (x1 … x16) |
| `V` | turbo: pencerenin izin verdiği en yüksek hız |
| `T` | sensör ışınları: kapalı → lider → tümü |
| `S` | modeli ve istatistikleri kaydet |
| `R` | yarışı yeniden başlat |
| `Esc` | kaydet ve menüye dön |

**En iyi modeli izle**, en hızlı turu atan modeli keşif kapalıyken tek arabayla gösterir.

### Pencere olmadan hızlı eğitim

```bash
python main.py --headless --episodes 300
```

```bash
python main.py --headless --resume --episodes 200
```

## Nasıl çalışıyor?

1. **Pist analizi** (`racing/track.py`): Koyu pikseller pist, kırmızı çizgi başlangıç/bitiştir.
   Elle checkpoint çizmeye gerek yok: başlangıç çizgisinden itibaren pist boyunca jeodezik mesafe
   (Dijkstra) hesaplanır. Bu "ilerleme alanı" hem ödülü hem de tur/ters yön tespitini sağlar.
2. **Ortam** (`racing/env.py`): Tüm arabalar numpy ile tek adımda güncellenir: ivme, fren,
   sürtünme, yüksek hızda savrulma, 7 mesafe sensörü. Zaman sabit adımlıdır; simülasyon hızı
   tur sürelerini ve öğrenmeyi etkilemez.
3. **Gözlem**: sensör mesafeleri, hız, arabanın pist yönüne göre açısı (sin/cos) ve son direksiyon.
4. **Ödül**: pist boyunca ilerleme (+), her adım küçük ceza, pistten çıkma / ters yön / hareketsizlik (−),
   tur tamamlama (+20).
5. **Ajan** (`racing/agent.py`): Dueling mimari, Double DQN hedefi, 3 adımlı getiri, yumuşak hedef ağ,
   deneyim tekrarı. Arabalar farklı keşif oranlarıyla sürer: ilk araba neredeyse açgözlüdür
   (öğrendiğini gösterir), sonuncusu en çok keşfeder.

## Rastgele pist oluşturma

Menüde **Pist oluştur**: `←`/`→` ile zorluk (**Kolay / Orta / Zor / Rastgele**), `R` ile yeni pist, `Enter`
ile kullan, `O` ile varsayılan piste dön. Pistler her seferinde farklıdır, kendini kesmez ve kapalı
döngüdür; `tracks/` klasörüne kaydedilir.

| Zorluk | Genişlik | Karakter |
|---|---|---|
| Kolay | 96–112 px | geniş, yumuşak virajlar |
| Orta | 72–84 px | sık virajlar |
| Zor | 54–64 px | dar, keskin virajlar ve şikanlar |

Komut satırından da üretilebilir (aynı tohum = aynı pist):

```bash
python main.py --new-track zor --seed 42
```

Mevcut model yeni pistte eğitime devam edebilir (sensörlerle genelleme); tur rekorları piste özeldir.

## Kendi pistinizi çizin

Herhangi bir resim programında:

- zemin **beyaz**, pist **siyah** (kapalı bir döngü olmalı),
- başlangıç/bitiş için pisti enlemesine kesen **kırmızı** (255, 0, 0) bir çizgi.

Dosyayı kaydedip `config.yaml` içinde `track.file` değerini değiştirin. Arabalar varsayılan olarak
çizgiden sağa (dikey çizgide aşağıya) doğru başlar; ters yön için `track.reverse: true`.
Kırmızı çizgi yoksa pistin en geniş noktası başlangıç kabul edilir.

## Ayarlar

Menüdeki **Ayarlar** ekranından ya da `config.yaml` üzerinden. Eksik alanlar varsayılanla tamamlanır,
sınır dışı değerler kırpılır. Sensör sayısı veya ağ boyutu değişirse kayıtlı model kullanılamaz
(yeni eğitim gerekir); uygulama bunu açık bir mesajla bildirir.

## Dosyalar

```
main.py            giriş noktası (menü veya --headless)
config.yaml        ayarlar
track.png          varsayılan pist
racing/
  track.py         pist analizi (maske, ilerleme ve yön alanları)
  trackgen.py      rastgele pist üretici
  picker.py        pist oluşturma ekranı
  env.py           vektörize fizik, sensörler, ödüller
  agent.py         DQN ajanı, deneyim havuzu
  trainer.py       eğitim/izleme döngüsü (arayüzden bağımsız)
  storage.py       istatistik, günlük ve model dosya yolları
  app.py           pencere, oturum döngüsü
  render.py        pist ve araba çizimi
  hud.py           yan bilgi paneli ve grafikler
  ui.py            menü, ayarlar, diyaloglar
  theme.py         renkler ve yazı tipleri
yurume.py          AI Walker giriş noktası
walker/
  model.py         hümanoid fizik modeli (MuJoCo MJCF)
  env.py           32 paralel karakter, ödül, düşme tespiti
  ppo.py           PPO öğrenme algoritması
  trainer.py       eğitim/izleme döngüsü, istatistikler
  render.py        yandan görünüm sahnesi ve karakter çizimi
  hud.py, app.py   panel ve pencere
tracks/            üretilen pistler
tests/             pytest testleri
data/              model.pt, best_model.pt, training_stats.json  (eğitimle oluşur)
logs/episodes.csv  bölüm günlüğü                                 (eğitimle oluşur)
_eski_surum/       önceki sürümün birebir yedeği
```

## Testler

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest --cov
```

## Eski sürümden farklar

Eski sürüm 3092 bölümde **hiç tur tamamlayamamıştı**. Başlıca nedenler ve çözümleri:

| Eski sorun | Yeni çözüm |
|---|---|
| Pistte checkpoint yoktu; otomatik checkpoint'ler merkez etrafında bir çembere, çoğu pist **dışına** konuyordu | Jeodezik ilerleme alanı: her pist görseliyle checkpoint'siz çalışır |
| DQN, çarpışma anlarını (terminal geçişleri) deneyim havuzundan **atıyordu**; ajan kaza yapmanın kötü olduğunu hiç öğrenemiyordu | Terminal ve zaman sınırı ayrımıyla doğru n-adımlı hedefler |
| Ödüller ve tur süreleri duvar saatine bağlıydı; hızlandırınca her şey bozuluyordu | Sabit zaman adımı |
| Adam için öğrenme oranı 0.101 (≈300 kat fazla) | 0.0003 |
| Sensörler 4 karede bir güncelleniyordu; ajan eski bilgiye göre karar veriyordu | Her adımda vektörize ışın izleme |
| 10 araba 10 ayrı ağ eğitiyordu, kaydedilen ise kullanılmayan başka bir ajandı | Tek ortak ağ; doğru model kaydediliyor, en iyi tur ayrıca saklanıyor |
| Pist (1336 px) pencereye (1200 px) sığmıyordu | Pist görüntü alanına ölçekleniyor |
| Ayarlar ekranında İptal, değişiklikleri yine de uyguluyordu | Taslak kopya üzerinde düzenleme |
