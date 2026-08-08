"""Klasik goruntu isleme oznitelikleri.

Her yaprak goruntusunden el ile tasarlanmis (hand-crafted) oznitelikler cikarir.
Uc grup oznitelik kullanilir:

1. Renk        - HSV ve Lab histogramlari, kanal bazli renk momentleri.
                 Hastaliklarin cogu renk degisimiyle belli oluyor: pas turuncu-
                 kahve lekeler, sararma sari, kirmizi curukluk kirmizi-bordo.
2. Doku        - LBP (Local Binary Pattern) histogrami ve GLCM istatistikleri.
                 Mozaik hastaligi renkten cok desen bozuklugu olarak goruluyor,
                 bu yuzden doku bilgisi renk kadar onemli.
3. Leke orani  - Yaprak maskesi icinde saglikli yesil disinda kalan piksellerin
                 orani. Hastaligin yayginligini tek sayiyla ozetler.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Tum goruntuler bu boyuta olceklenir; hiz ile ayrinti arasinda denge
HEDEF_BOYUT = (256, 256)

# LBP ayarlari: 8 komsu, 1 piksel yaricap, donmeye duyarsiz tekduze desenler
LBP_KOMSU = 8
LBP_YARICAP = 1
LBP_KOVA_SAYISI = LBP_KOMSU + 2  # "uniform" yontemi bu kadar kova uretir


def goruntu_oku(yol):
    """Goruntuyu diskten okur ve sabit boyuta olcekler (BGR duzeninde)."""
    goruntu = cv2.imread(str(yol), cv2.IMREAD_COLOR)
    if goruntu is None:
        raise ValueError(f"Goruntu okunamadi: {yol}")
    return cv2.resize(goruntu, HEDEF_BOYUT, interpolation=cv2.INTER_AREA)


def yaprak_maskesi(goruntu_bgr):
    """Arka plandan yapragi ayiran ikili maske uretir.

    Otsu esiklemesi doygunluk (saturation) kanalina uygulanir; arka plan
    genelde dusuk doygunlukta oldugu icin yaprak bu sekilde iyi ayrisiyor.
    """
    hsv = cv2.cvtColor(goruntu_bgr, cv2.COLOR_BGR2HSV)
    doygunluk = hsv[:, :, 1]

    _, maske = cv2.threshold(doygunluk, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kucuk deliklerin ve gurultunun temizlenmesi
    cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, cekirdek)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, cekirdek)

    # Maske neredeyse bosalirsa esikleme basarisiz olmus demektir; tumunu kullan
    if maske.mean() < 10:
        maske = np.full(maske.shape, 255, dtype=np.uint8)

    return maske


def _histogram(kanal, maske, kova_sayisi, ust_sinir):
    """Tek kanal icin normalize edilmis histogram dondurur."""
    hist = cv2.calcHist([kanal], [0], maske, [kova_sayisi], [0, ust_sinir])
    hist = hist.flatten()
    toplam = hist.sum()
    return hist / toplam if toplam > 0 else hist


def _renk_momentleri(kanal, maske_bool):
    """Bir kanalin ortalama, standart sapma ve carpikligini dondurur."""
    degerler = kanal[maske_bool].astype(np.float64)
    if degerler.size == 0:
        return [0.0, 0.0, 0.0]

    ortalama = degerler.mean()
    std = degerler.std()
    # Carpiklik (skewness); std sifirsa tanimsiz oldugu icin 0 verilir
    carpiklik = 0.0 if std < 1e-6 else float(((degerler - ortalama) ** 3).mean() / std ** 3)
    return [float(ortalama), float(std), carpiklik]


def renk_oznitelikleri(goruntu_bgr, maske):
    """HSV + Lab histogramlari ve renk momentlerini birlestirir."""
    hsv = cv2.cvtColor(goruntu_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(goruntu_bgr, cv2.COLOR_BGR2LAB)
    maske_bool = maske > 0

    parcalar = [
        _histogram(hsv[:, :, 0], maske, 32, 180),  # renk tonu
        _histogram(hsv[:, :, 1], maske, 32, 256),  # doygunluk
        _histogram(hsv[:, :, 2], maske, 32, 256),  # parlaklik
        _histogram(lab[:, :, 1], maske, 32, 256),  # yesil-kirmizi ekseni
        _histogram(lab[:, :, 2], maske, 32, 256),  # mavi-sari ekseni
    ]

    momentler = []
    for kaynak in (goruntu_bgr, hsv, lab):
        for k in range(3):
            momentler.extend(_renk_momentleri(kaynak[:, :, k], maske_bool))

    return np.concatenate(parcalar + [np.array(momentler, dtype=np.float64)])


def doku_oznitelikleri(goruntu_bgr, maske):
    """LBP histogrami ve GLCM istatistiklerini dondurur."""
    gri = cv2.cvtColor(goruntu_bgr, cv2.COLOR_BGR2GRAY)

    # --- LBP ---
    lbp = local_binary_pattern(gri, LBP_KOMSU, LBP_YARICAP, method="uniform")
    lbp_degerler = lbp[maske > 0]
    if lbp_degerler.size == 0:
        lbp_hist = np.zeros(LBP_KOVA_SAYISI)
    else:
        lbp_hist, _ = np.histogram(
            lbp_degerler, bins=LBP_KOVA_SAYISI, range=(0, LBP_KOVA_SAYISI)
        )
        lbp_hist = lbp_hist.astype(np.float64)
        lbp_hist /= lbp_hist.sum()

    # --- GLCM ---
    # 32 seviyeye indirgemek matrisi kucultup gurultuyu azaltir
    gri_indirgenmis = (gri / 8).astype(np.uint8)
    glcm = graycomatrix(
        gri_indirgenmis,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )

    glcm_degerleri = []
    for ozellik in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        glcm_degerleri.extend(graycoprops(glcm, ozellik).flatten())

    return np.concatenate([lbp_hist, np.array(glcm_degerleri, dtype=np.float64)])


def leke_orani(goruntu_bgr, maske):
    """Yaprak icinde saglikli yesil disinda kalan piksellerin oranini olcer."""
    hsv = cv2.cvtColor(goruntu_bgr, cv2.COLOR_BGR2HSV)
    maske_bool = maske > 0
    yaprak_piksel = int(maske_bool.sum())
    if yaprak_piksel == 0:
        return np.zeros(3)

    ton = hsv[:, :, 0]
    doygunluk = hsv[:, :, 1]

    # OpenCV'de ton 0-179 arasindadir. Yesil yaklasik 35-85 araligina duser.
    yesil = maske_bool & (ton >= 35) & (ton <= 85) & (doygunluk > 40)
    # Sari-turuncu bolge: sararma ve pas belirtileri
    sari = maske_bool & (ton >= 20) & (ton < 35)
    # Kirmizi-kahve bolge: kirmizi curukluk belirtileri (ton ekseninin iki ucu)
    kirmizi = maske_bool & ((ton < 20) | (ton > 160))

    return np.array([
        1.0 - yesil.sum() / yaprak_piksel,  # saglikli olmayan alan orani
        sari.sum() / yaprak_piksel,
        kirmizi.sum() / yaprak_piksel,
    ])


def oznitelik_cikar(yol):
    """Tek bir goruntu icin tum oznitelikleri tek vektorde birlestirir."""
    goruntu = goruntu_oku(yol)
    maske = yaprak_maskesi(goruntu)

    return np.concatenate([
        renk_oznitelikleri(goruntu, maske),
        doku_oznitelikleri(goruntu, maske),
        leke_orani(goruntu, maske),
    ]).astype(np.float32)


def toplu_oznitelik_cikar(yollar, ilerleme_adimi=250):
    """Verilen yollarin tamami icin oznitelik matrisi uretir."""
    oznitelikler = []
    for sira, yol in enumerate(yollar, start=1):
        oznitelikler.append(oznitelik_cikar(yol))
        if ilerleme_adimi and sira % ilerleme_adimi == 0:
            print(f"  {sira}/{len(yollar)} goruntu islendi")
    return np.vstack(oznitelikler)


if __name__ == "__main__":
    from veri import goruntuleri_topla

    yollar, _ = goruntuleri_topla()
    vektor = oznitelik_cikar(yollar[0])
    print(f"Ornek goruntu : {yollar[0].name}")
    print(f"Oznitelik boyu: {vektor.shape[0]}")
