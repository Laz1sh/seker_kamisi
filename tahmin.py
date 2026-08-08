"""Tek bir yaprak fotografini siniflandirir.

Egitilmis modellerden birini yukleyip verilen goruntu icin hastalik tahmini
ve guven skorlarini yazdirir.

Kullanim:
    python tahmin.py fotograf.jpeg
    python tahmin.py fotograf.jpeg --model klasik
"""

import argparse
import sys
from pathlib import Path

import numpy as np

KOK_DIZIN = Path(__file__).resolve().parent
sys.path.insert(0, str(KOK_DIZIN / "src"))

from veri import SINIFLAR, SINIF_ADLARI_TR  # noqa: E402

KLASIK_MODEL = KOK_DIZIN / "sonuclar" / "klasik_model.joblib"
DERIN_MODEL = KOK_DIZIN / "sonuclar" / "derin_model.keras"

# Hastaliklarin kisa aciklamalari - ciktiyi tarim kullanicisi icin anlamli kilar
ACIKLAMALAR = {
    "Healthy": "Yaprakta hastalik belirtisi gorulmedi.",
    "Mosaic": "Mozaik virusu: yaprakta acik-koyu yesil benekli desen.",
    "RedRot": "Kirmizi curukluk: mantar kaynakli, kirmizi-bordo lezyonlar.",
    "Rust": "Pas hastaligi: turuncu-kahverengi kabarcik seklinde lekeler.",
    "Yellow": "Sararma: yaprak damarlari boyunca sari renk kaybi.",
}


def klasik_tahmin(goruntu_yolu):
    import joblib
    from oznitelik import oznitelik_cikar

    if not KLASIK_MODEL.exists():
        raise FileNotFoundError(
            "Klasik model bulunamadi. Once 'python src/klasik.py' calistirin."
        )

    paket = joblib.load(KLASIK_MODEL)
    model = paket["model"]
    oznitelikler = oznitelik_cikar(goruntu_yolu).reshape(1, -1)

    # SVC olasilik uretmiyorsa karar fonksiyonundan yumusak skor turetilir
    if hasattr(model, "predict_proba"):
        olasiliklar = model.predict_proba(oznitelikler)[0]
    else:
        skorlar = model.decision_function(oznitelikler)[0]
        ussel = np.exp(skorlar - skorlar.max())
        olasiliklar = ussel / ussel.sum()

    return olasiliklar, paket["model_adi"]


def derin_tahmin(goruntu_yolu):
    import tensorflow as tf

    if not DERIN_MODEL.exists():
        raise FileNotFoundError(
            "Derin model bulunamadi. Once 'python src/derin.py' calistirin."
        )

    model = tf.keras.models.load_model(DERIN_MODEL)

    ham = tf.io.read_file(str(goruntu_yolu))
    goruntu = tf.image.decode_image(ham, channels=3, expand_animations=False)
    goruntu = tf.image.resize(goruntu, (224, 224))
    goruntu = tf.keras.applications.mobilenet_v2.preprocess_input(goruntu)

    olasiliklar = model.predict(tf.expand_dims(goruntu, 0), verbose=0)[0]
    return olasiliklar, "MobileNetV2"


def main():
    ayristirici = argparse.ArgumentParser(
        description="Seker kamisi yaprak hastaligi tahmini"
    )
    ayristirici.add_argument("goruntu", help="Siniflandirilacak fotograf yolu")
    ayristirici.add_argument(
        "--model", choices=["derin", "klasik"], default="derin",
        help="Kullanilacak model (varsayilan: derin)",
    )
    argumanlar = ayristirici.parse_args()

    goruntu_yolu = Path(argumanlar.goruntu)
    if not goruntu_yolu.exists():
        ayristirici.error(f"Dosya bulunamadi: {goruntu_yolu}")

    if argumanlar.model == "klasik":
        olasiliklar, model_adi = klasik_tahmin(goruntu_yolu)
    else:
        olasiliklar, model_adi = derin_tahmin(goruntu_yolu)

    sira = np.argsort(olasiliklar)[::-1]
    kazanan = SINIFLAR[sira[0]]

    print(f"\nDosya : {goruntu_yolu.name}")
    print(f"Model : {model_adi}")
    print(f"\nTahmin: {SINIF_ADLARI_TR[kazanan]}  (guven: %{olasiliklar[sira[0]]*100:.1f})")
    print(f"        {ACIKLAMALAR[kazanan]}\n")

    print("Tum siniflar:")
    for indeks in sira:
        sinif = SINIFLAR[indeks]
        oran = olasiliklar[indeks] * 100
        cubuk = "#" * int(round(oran / 4))
        print(f"  {SINIF_ADLARI_TR[sinif]:<18} %{oran:5.1f}  {cubuk}")


if __name__ == "__main__":
    main()
