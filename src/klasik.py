"""Klasik goruntu isleme + makine ogrenmesi yaklasimi.

El ile tasarlanmis renk / doku / leke oznitelikleri cikarilir, ardindan uc
farkli siniflandirici egitilip dogrulama kumesinde karsilastirilir. En iyi
model test kumesinde raporlanir.

Kullanim:
    python src/klasik.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from degerlendir import CIKTI_DIZIN, cikti_dizini_hazirla, rapor_uret
from oznitelik import oznitelik_cikar
from veri import goruntuleri_topla, veriyi_bol, dagilimi_yazdir, RASGELE_TOHUM

KOK_DIZIN = Path(__file__).resolve().parent.parent
ONBELLEK = KOK_DIZIN / "sonuclar" / "oznitelikler.npz"
MODEL_DOSYASI = KOK_DIZIN / "sonuclar" / "klasik_model.joblib"


def oznitelikleri_hazirla(bolunmus, yeniden_hesapla=False):
    """Tum kumeler icin oznitelik matrislerini uretir; sonucu diske onbellekler.

    Oznitelik cikarimi birkac dakika surdugu icin bir kez hesaplanip
    saklanir. Onbellek varsa dogrudan okunur.
    """
    cikti_dizini_hazirla()

    if ONBELLEK.exists() and not yeniden_hesapla:
        print(f"Onbellekten okunuyor: {ONBELLEK.name}")
        veri = np.load(ONBELLEK)
        return {
            kume: (veri[f"{kume}_X"], veri[f"{kume}_y"])
            for kume in ("egitim", "dogrulama", "test")
        }

    print("Oznitelikler cikariliyor (tum cekirdekler kullaniliyor)...")
    baslangic = time.time()
    sonuc = {}

    for kume, (yollar, etiketler) in bolunmus.items():
        print(f"  [{kume}] {len(yollar)} goruntu")
        oznitelikler = joblib.Parallel(n_jobs=-1, batch_size=32)(
            joblib.delayed(oznitelik_cikar)(yol) for yol in yollar
        )
        sonuc[kume] = (np.vstack(oznitelikler), etiketler)

    sure = time.time() - baslangic
    print(f"Oznitelik cikarimi tamamlandi: {sure:.1f} saniye")

    np.savez_compressed(
        ONBELLEK,
        **{f"{k}_X": v[0] for k, v in sonuc.items()},
        **{f"{k}_y": v[1] for k, v in sonuc.items()},
    )
    print(f"Onbellege yazildi: {ONBELLEK.name}")
    return sonuc


def modelleri_tanimla():
    """Karsilastirilacak siniflandiricilar.

    Hepsi StandardScaler ile ayni on isleme tabi tutulur; oznitelikler
    farkli olceklerde oldugu icin (histogram 0-1, GLCM kontrast yuzler
    mertebesinde) olcekleme sart.
    """
    return {
        "SVM (RBF cekirdek)": Pipeline([
            ("olcekle", StandardScaler()),
            ("model", SVC(C=10.0, gamma="scale", kernel="rbf",
                          random_state=RASGELE_TOHUM)),
        ]),
        "Rastgele Orman": Pipeline([
            ("olcekle", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                             random_state=RASGELE_TOHUM)),
        ]),
        "Lojistik Regresyon": Pipeline([
            ("olcekle", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0,
                                         random_state=RASGELE_TOHUM)),
        ]),
    }


def main():
    yollar, etiketler = goruntuleri_topla()
    print(f"Toplam goruntu: {len(yollar)}\n")

    bolunmus = veriyi_bol(yollar, etiketler)
    dagilimi_yazdir(bolunmus)
    print()

    oznitelikler = oznitelikleri_hazirla(bolunmus)
    egitim_X, egitim_y = oznitelikler["egitim"]
    dogrulama_X, dogrulama_y = oznitelikler["dogrulama"]
    test_X, test_y = oznitelikler["test"]
    print(f"\nOznitelik vektoru boyutu: {egitim_X.shape[1]}\n")

    # --- Dogrulama kumesinde model secimi ---
    print("Modeller egitiliyor ve dogrulama kumesinde karsilastiriliyor:")
    skorlar = {}
    egitilmis = {}

    for ad, model in modelleri_tanimla().items():
        baslangic = time.time()
        model.fit(egitim_X, egitim_y)
        skor = accuracy_score(dogrulama_y, model.predict(dogrulama_X))
        skorlar[ad] = skor
        egitilmis[ad] = model
        print(f"  {ad:<22} dogrulama dogrulugu: {skor:.4f}  ({time.time()-baslangic:.1f} sn)")

    en_iyi_ad = max(skorlar, key=skorlar.get)
    print(f"\nSecilen model: {en_iyi_ad}")

    # --- Test kumesinde nihai degerlendirme ---
    en_iyi_model = egitilmis[en_iyi_ad]
    test_tahmin = en_iyi_model.predict(test_X)
    rapor_uret(test_y, test_tahmin, f"Klasik - {en_iyi_ad}", "klasik")

    joblib.dump({"model": en_iyi_model, "model_adi": en_iyi_ad}, MODEL_DOSYASI)
    print(f"Model kaydedildi: {MODEL_DOSYASI.relative_to(KOK_DIZIN)}")
    print(f"Grafikler ve rapor: {CIKTI_DIZIN.relative_to(KOK_DIZIN)}/")


if __name__ == "__main__":
    main()
