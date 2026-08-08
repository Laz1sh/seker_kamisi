"""Iki yaklasimin sonuclarini yan yana karsilastirir.

klasik.py ve derin.py calistirildiktan sonra uretilen JSON raporlarini okur,
karsilastirma tablosunu ekrana basar ve sinif bazli F1 skorlarini tek grafikte
gorsellestirir.

Kullanim:
    python src/karsilastir.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from degerlendir import CIKTI_DIZIN, turkce_etiketler

KOK_DIZIN = Path(__file__).resolve().parent.parent
GRAFIK_DOSYASI = CIKTI_DIZIN / "karsilastirma.png"
OZET_DOSYASI = CIKTI_DIZIN / "karsilastirma.md"


def rapor_oku(dosya_onEki):
    yol = CIKTI_DIZIN / f"{dosya_onEki}_rapor.json"
    if not yol.exists():
        return None
    return json.loads(yol.read_text(encoding="utf-8"))


def grafik_ciz(raporlar):
    """Sinif bazli F1 skorlarini gruplu cubuk grafikte gosterir."""
    etiketler = turkce_etiketler()
    konumlar = np.arange(len(etiketler))
    genislik = 0.8 / len(raporlar)

    fig, eksen = plt.subplots(figsize=(10, 5.5))

    for sira, rapor in enumerate(raporlar):
        skorlar = [rapor["sinif_bazli"][etiket]["f1-score"] for etiket in etiketler]
        kaydirma = (sira - (len(raporlar) - 1) / 2) * genislik
        cubuklar = eksen.bar(konumlar + kaydirma, skorlar, genislik,
                             label=rapor["model"])
        eksen.bar_label(cubuklar, fmt="%.2f", fontsize=8, padding=2)

    eksen.set_xticks(konumlar)
    eksen.set_xticklabels(etiketler, rotation=15, ha="right")
    eksen.set_ylabel("F1 skoru")
    eksen.set_ylim(0, 1.08)
    eksen.set_title("Sinif Bazli F1 Skorlari - Yaklasim Karsilastirmasi")
    eksen.legend(loc="lower right")
    eksen.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(GRAFIK_DOSYASI, dpi=150)
    plt.close(fig)


def ozet_yaz(raporlar):
    """Karsilastirma tablosunu Markdown olarak kaydeder."""
    etiketler = turkce_etiketler()
    satirlar = ["# Sonuc Karsilastirmasi", "", "## Genel", "",
                "| Model | Dogruluk | Makro F1 |", "|---|---|---|"]

    for rapor in raporlar:
        satirlar.append(
            f"| {rapor['model']} | %{rapor['dogruluk']*100:.2f} | {rapor['makro_f1']:.4f} |"
        )

    satirlar += ["", "## Sinif bazli F1", "",
                 "| Sinif | " + " | ".join(r["model"] for r in raporlar) + " |",
                 "|---" * (len(raporlar) + 1) + "|"]

    for etiket in etiketler:
        degerler = [f"{r['sinif_bazli'][etiket]['f1-score']:.3f}" for r in raporlar]
        satirlar.append(f"| {etiket} | " + " | ".join(degerler) + " |")

    OZET_DOSYASI.write_text("\n".join(satirlar) + "\n", encoding="utf-8")


def main():
    raporlar = [r for r in (rapor_oku("klasik"), rapor_oku("derin")) if r]

    if not raporlar:
        print("Rapor bulunamadi. Once src/klasik.py ve src/derin.py calistirin.")
        return
    if len(raporlar) == 1:
        print(f"Yalnizca bir rapor var ({raporlar[0]['model']}); "
              "karsilastirma icin digerini de calistirin.\n")

    baslik = f"{'Model':<34}{'Dogruluk':>12}{'Makro F1':>12}"
    print(baslik)
    print("-" * len(baslik))
    for rapor in raporlar:
        print(f"{rapor['model']:<34}{rapor['dogruluk']*100:>11.2f}%{rapor['makro_f1']:>12.4f}")

    if len(raporlar) > 1:
        fark = (raporlar[1]["dogruluk"] - raporlar[0]["dogruluk"]) * 100
        yon = "yuksek" if fark >= 0 else "dusuk"
        print(f"\nFark: derin ogrenme, klasik yaklasima gore {abs(fark):.2f} puan {yon}.")

    grafik_ciz(raporlar)
    ozet_yaz(raporlar)
    print(f"\nGrafik : {GRAFIK_DOSYASI.relative_to(KOK_DIZIN)}")
    print(f"Ozet   : {OZET_DOSYASI.relative_to(KOK_DIZIN)}")


if __name__ == "__main__":
    main()
