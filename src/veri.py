"""Veri seti yukleme ve bolme islemleri.

Depodaki goruntuler sinif adiyla adlandirilmis klasorlerde duruyor:
    Healthy/ Mosaic/ RedRot/ Rust/ Yellow/

Bu modul goruntu yollarini ve etiketlerini toplar, ardindan egitim / dogrulama
/ test kumelerine sinif dagilimini koruyarak (stratified) boler.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Proje kok dizini: bu dosya src/ altinda oldugu icin bir ust klasor
KOK_DIZIN = Path(__file__).resolve().parent.parent

SINIFLAR = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

# Rapor ve grafiklerde kullanilacak okunabilir Turkce karsiliklar
SINIF_ADLARI_TR = {
    "Healthy": "Saglikli",
    "Mosaic": "Mozaik",
    "RedRot": "Kirmizi Curukluk",
    "Rust": "Pas",
    "Yellow": "Sararma",
}

RASGELE_TOHUM = 42


def goruntuleri_topla(kok_dizin=KOK_DIZIN):
    """Tum goruntu yollarini ve sayisal etiketlerini dondurur.

    Returns:
        yollar: (N,) numpy dizisi, Path nesneleri
        etiketler: (N,) numpy dizisi, 0..4 arasi sinif indeksleri
    """
    yollar = []
    etiketler = []

    for indeks, sinif in enumerate(SINIFLAR):
        sinif_dizini = Path(kok_dizin) / sinif
        if not sinif_dizini.is_dir():
            raise FileNotFoundError(f"Sinif klasoru bulunamadi: {sinif_dizini}")

        # Buyuk/kucuk harf farkina takilmamak icin tum uzantilari tara
        bulunanlar = sorted(
            p for p in sinif_dizini.iterdir()
            if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        if not bulunanlar:
            raise FileNotFoundError(f"{sinif_dizini} icinde goruntu yok")

        yollar.extend(bulunanlar)
        etiketler.extend([indeks] * len(bulunanlar))

    return np.array(yollar), np.array(etiketler)


def veriyi_bol(yollar, etiketler, test_orani=0.15, dogrulama_orani=0.15):
    """Veriyi egitim / dogrulama / test olarak sinif dengesini koruyarak boler.

    Varsayilan dagilim: %70 egitim, %15 dogrulama, %15 test.
    """
    # Once test kumesini ayir
    egitim_yol, test_yol, egitim_etiket, test_etiket = train_test_split(
        yollar,
        etiketler,
        test_size=test_orani,
        stratify=etiketler,
        random_state=RASGELE_TOHUM,
    )

    # Kalan kisimdan dogrulama kumesini ayir.
    # dogrulama_orani tum veriye gore verildigi icin kalan kumeye gore olceklenir.
    kalan_oran = dogrulama_orani / (1.0 - test_orani)
    egitim_yol, dogrulama_yol, egitim_etiket, dogrulama_etiket = train_test_split(
        egitim_yol,
        egitim_etiket,
        test_size=kalan_oran,
        stratify=egitim_etiket,
        random_state=RASGELE_TOHUM,
    )

    return {
        "egitim": (egitim_yol, egitim_etiket),
        "dogrulama": (dogrulama_yol, dogrulama_etiket),
        "test": (test_yol, test_etiket),
    }


def dagilimi_yazdir(bolunmus):
    """Kume basina sinif dagilimini tabloya benzer bicimde ekrana basar."""
    baslik = f"{'Kume':<12}" + "".join(f"{s:>12}" for s in SINIFLAR) + f"{'Toplam':>10}"
    print(baslik)
    print("-" * len(baslik))

    for kume_adi, (_, etiketler) in bolunmus.items():
        sayimlar = [int(np.sum(etiketler == i)) for i in range(len(SINIFLAR))]
        satir = f"{kume_adi:<12}" + "".join(f"{c:>12}" for c in sayimlar)
        print(satir + f"{len(etiketler):>10}")


if __name__ == "__main__":
    yollar, etiketler = goruntuleri_topla()
    print(f"Toplam goruntu: {len(yollar)}\n")
    dagilimi_yazdir(veriyi_bol(yollar, etiketler))
