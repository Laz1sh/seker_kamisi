"""Ortak degerlendirme yardimcilari.

Hem klasik hem derin ogrenme modelleri ayni olcutlerle raporlanir ki
karsilastirma adil olsun.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ekran olmadan dosyaya cizim yapabilmek icin

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from veri import SINIF_ADLARI_TR, SINIFLAR

KOK_DIZIN = Path(__file__).resolve().parent.parent
CIKTI_DIZIN = KOK_DIZIN / "sonuclar"


def cikti_dizini_hazirla():
    CIKTI_DIZIN.mkdir(exist_ok=True)
    return CIKTI_DIZIN


def turkce_etiketler():
    return [SINIF_ADLARI_TR[s] for s in SINIFLAR]


def karisiklik_matrisi_ciz(gercek, tahmin, baslik, dosya_adi):
    """Karisiklik matrisini yuzde degerleriyle birlikte gorsellestirir."""
    cikti_dizini_hazirla()
    matris = confusion_matrix(gercek, tahmin)
    etiketler = turkce_etiketler()

    fig, eksen = plt.subplots(figsize=(7.5, 6.5))
    goruntu = eksen.imshow(matris, cmap="Blues")

    eksen.set_xticks(range(len(etiketler)))
    eksen.set_yticks(range(len(etiketler)))
    eksen.set_xticklabels(etiketler, rotation=30, ha="right")
    eksen.set_yticklabels(etiketler)
    eksen.set_xlabel("Modelin tahmini")
    eksen.set_ylabel("Gercek sinif")
    eksen.set_title(baslik)

    # Hucre icine hem adet hem satir yuzdesi yaz
    satir_toplam = matris.sum(axis=1, keepdims=True)
    yuzdeler = np.divide(
        matris, satir_toplam,
        out=np.zeros(matris.shape, dtype=np.float64),
        where=satir_toplam > 0,
    ) * 100
    esik = matris.max() / 2.0

    for i in range(matris.shape[0]):
        for j in range(matris.shape[1]):
            eksen.text(
                j, i,
                f"{matris[i, j]}\n%{yuzdeler[i, j]:.0f}",
                ha="center", va="center",
                color="white" if matris[i, j] > esik else "black",
                fontsize=9,
            )

    fig.colorbar(goruntu, ax=eksen, shrink=0.8)
    fig.tight_layout()
    hedef = CIKTI_DIZIN / dosya_adi
    fig.savefig(hedef, dpi=150)
    plt.close(fig)
    return hedef


def rapor_uret(gercek, tahmin, model_adi, dosya_onEki):
    """Olcutleri hesaplar, ekrana basar ve JSON olarak kaydeder."""
    cikti_dizini_hazirla()
    etiketler = turkce_etiketler()

    dogruluk = accuracy_score(gercek, tahmin)
    makro_f1 = f1_score(gercek, tahmin, average="macro")

    print(f"\n=== {model_adi} - test kumesi sonuclari ===")
    print(f"Dogruluk (accuracy): {dogruluk:.4f}")
    print(f"Makro F1           : {makro_f1:.4f}\n")
    print(classification_report(gercek, tahmin, target_names=etiketler, digits=3))

    ayrinti = classification_report(
        gercek, tahmin, target_names=etiketler, output_dict=True, digits=4
    )
    ozet = {
        "model": model_adi,
        "dogruluk": float(dogruluk),
        "makro_f1": float(makro_f1),
        "sinif_bazli": ayrinti,
    }

    hedef = CIKTI_DIZIN / f"{dosya_onEki}_rapor.json"
    hedef.write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")

    karisiklik_matrisi_ciz(
        gercek, tahmin,
        baslik=f"{model_adi} - Karisiklik Matrisi",
        dosya_adi=f"{dosya_onEki}_karisiklik_matrisi.png",
    )

    return ozet
