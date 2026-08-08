"""Derin ogrenme yaklasimi - MobileNetV2 ile transfer ogrenme.

ImageNet uzerinde onceden egitilmis MobileNetV2 govdesi ozniteli cikarici
olarak kullanilir. Iki asamali egitim uygulanir:

1. Isinma  - Govde tamamen dondurulur, yalnizca yeni siniflandirma katmani
             egitilir. Rastgele baslayan katmanin buyuk gradyanlarla onceden
             ogrenilmis agirliklari bozmasi bu sekilde onlenir.
2. Ince ayar - Govdenin son bloklari cozulup cok dusuk ogrenme oraniyla
             birlikte egitilir; model seker kamisi yapraklarina uyarlanir.

CPU uzerinde calisacak sekilde ayarlandi (goruntu 224x224, yigin boyu 32).

Kullanim:
    python src/derin.py
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from degerlendir import CIKTI_DIZIN, cikti_dizini_hazirla, rapor_uret
from veri import SINIFLAR, goruntuleri_topla, veriyi_bol, dagilimi_yazdir, RASGELE_TOHUM

KOK_DIZIN = Path(__file__).resolve().parent.parent
MODEL_DOSYASI = KOK_DIZIN / "sonuclar" / "derin_model.keras"
GECMIS_DOSYASI = KOK_DIZIN / "sonuclar" / "derin_egitim_grafigi.png"

GORUNTU_BOYUTU = (224, 224)
YIGIN_BOYU = 32
ISINMA_DONGUSU = 10
INCE_AYAR_DONGUSU = 15

# Ince ayarda govdenin kacinci katmandan sonrasi cozulecek
COZULME_KATMANI = 100

tf.keras.utils.set_random_seed(RASGELE_TOHUM)


def veri_kumesi_kur(yollar, etiketler, karistir=False, artir=False):
    """Verilen yol listesinden bir tf.data veri hatti olusturur."""
    yol_metinleri = [str(p) for p in yollar]
    veri = tf.data.Dataset.from_tensor_slices((yol_metinleri, etiketler))

    if karistir:
        veri = veri.shuffle(len(yol_metinleri), seed=RASGELE_TOHUM,
                            reshuffle_each_iteration=True)

    def oku(yol, etiket):
        ham = tf.io.read_file(yol)
        goruntu = tf.image.decode_jpeg(ham, channels=3)
        goruntu = tf.image.resize(goruntu, GORUNTU_BOYUTU)
        return goruntu, etiket

    veri = veri.map(oku, num_parallel_calls=tf.data.AUTOTUNE)
    veri = veri.batch(YIGIN_BOYU)

    if artir:
        veri = veri.map(lambda g, e: (artirma_katmani(g, training=True), e),
                        num_parallel_calls=tf.data.AUTOTUNE)

    # MobileNetV2 kendi on islemesini bekler: pikselleri [-1, 1] araligina tasir
    veri = veri.map(
        lambda g, e: (tf.keras.applications.mobilenet_v2.preprocess_input(g), e),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return veri.prefetch(tf.data.AUTOTUNE)


# Veri artirma: yaprak fotograflari her acidan cekilebildigi icin cevirme ve
# donme mantikli; renk kaydirmasi hastalik rengini bozmayacak kadar hafif tutuldu.
artirma_katmani = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
    tf.keras.layers.RandomContrast(0.10),
], name="veri_artirma")


def model_kur():
    """MobileNetV2 govdesi + yeni siniflandirma basligi."""
    govde = tf.keras.applications.MobileNetV2(
        input_shape=GORUNTU_BOYUTU + (3,),
        include_top=False,
        weights="imagenet",
    )
    govde.trainable = False

    girdi = tf.keras.Input(shape=GORUNTU_BOYUTU + (3,))
    x = govde(girdi, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    cikti = tf.keras.layers.Dense(len(SINIFLAR), activation="softmax")(x)

    return tf.keras.Model(girdi, cikti), govde


def egitim_grafigi_ciz(gecmisler):
    """Isinma ve ince ayar asamalarinin dogruluk / kayip egrilerini cizer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    egitim_dogruluk, dogrulama_dogruluk = [], []
    egitim_kayip, dogrulama_kayip = [], []

    for gecmis in gecmisler:
        egitim_dogruluk += gecmis.history["accuracy"]
        dogrulama_dogruluk += gecmis.history["val_accuracy"]
        egitim_kayip += gecmis.history["loss"]
        dogrulama_kayip += gecmis.history["val_loss"]

    gecis_noktasi = len(gecmisler[0].history["accuracy"])
    fig, eksenler = plt.subplots(1, 2, figsize=(12, 4.5))

    for eksen, (egitim, dogrulama, baslik) in zip(
        eksenler,
        [(egitim_dogruluk, dogrulama_dogruluk, "Dogruluk"),
         (egitim_kayip, dogrulama_kayip, "Kayip")],
    ):
        eksen.plot(egitim, label="Egitim")
        eksen.plot(dogrulama, label="Dogrulama")
        eksen.axvline(gecis_noktasi - 0.5, color="gray", linestyle="--",
                      label="Ince ayar baslangici")
        eksen.set_xlabel("Dongu (epoch)")
        eksen.set_title(baslik)
        eksen.legend()
        eksen.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(GECMIS_DOSYASI, dpi=150)
    plt.close(fig)


def main():
    cikti_dizini_hazirla()

    yollar, etiketler = goruntuleri_topla()
    print(f"Toplam goruntu: {len(yollar)}\n")

    bolunmus = veriyi_bol(yollar, etiketler)
    dagilimi_yazdir(bolunmus)
    print()

    egitim = veri_kumesi_kur(*bolunmus["egitim"], karistir=True, artir=True)
    dogrulama = veri_kumesi_kur(*bolunmus["dogrulama"])
    test = veri_kumesi_kur(*bolunmus["test"])

    model, govde = model_kur()

    # --- 1. asama: isinma ---
    print("1. asama - siniflandirma basligi egitiliyor (govde donduruldu)")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    baslangic = time.time()
    isinma = model.fit(egitim, validation_data=dogrulama,
                       epochs=ISINMA_DONGUSU, verbose=2)

    # --- 2. asama: ince ayar ---
    print("\n2. asama - govdenin son bloklari cozulup ince ayar yapiliyor")
    govde.trainable = True

    for sira, katman in enumerate(govde.layers):
        # Ilk katmanlar genel kenar/doku bilgisi tasir, dokunmuyoruz
        if sira < COZULME_KATMANI:
            katman.trainable = False
        # BatchNormalization katmanlari donuk kalmali. Cozuldugunde kucuk
        # yiginlarla hesaplanan yeni ortalama/varyans degerleri onceden
        # ogrenilmis istatistikleri bozuyor ve egitim dogrulugu cokuyor.
        elif isinstance(katman, tf.keras.layers.BatchNormalization):
            katman.trainable = False

    cozulen = sum(1 for k in govde.layers if k.trainable)
    print(f"  Cozulen katman sayisi: {cozulen}/{len(govde.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ince_ayar = model.fit(
        egitim, validation_data=dogrulama, epochs=INCE_AYAR_DONGUSU, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
        ],
    )
    print(f"\nToplam egitim suresi: {time.time() - baslangic:.0f} saniye")

    # --- Test kumesinde degerlendirme ---
    olasiliklar = model.predict(test, verbose=0)
    tahmin = np.argmax(olasiliklar, axis=1)
    gercek = np.concatenate([e.numpy() for _, e in test])

    rapor_uret(gercek, tahmin, "Derin Ogrenme - MobileNetV2", "derin")

    egitim_grafigi_ciz([isinma, ince_ayar])
    model.save(MODEL_DOSYASI)
    print(f"Model kaydedildi: {MODEL_DOSYASI.relative_to(KOK_DIZIN)}")
    print(f"Grafikler ve rapor: {CIKTI_DIZIN.relative_to(KOK_DIZIN)}/")


if __name__ == "__main__":
    main()
