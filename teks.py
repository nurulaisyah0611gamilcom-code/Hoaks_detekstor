"""Pembersihan teks + fungsi prediksi. Dipakai notebook, training, dan web app (satu sumber kebenaran)."""
import re
import pandas as pd

MIN_KATA = 5   # teks yang lebih pendek dari ini tidak dinilai

def ambil_sumber(url):
    """Ambil nama situs dari URL, misalnya 'turnbackhoax.id'."""
    m = re.match(r"https?://(?:www\.)?([^/]+)", str(url))
    return m.group(1) if m else "tidak_diketahui"

def clean_text(t):
    """Hapus penanda sumber berita, sisa tampilan situs web, URL, angka, dan tanda baca; ubah ke huruf kecil."""
    t = str(t)
    t = re.sub(r"^\s*[A-Za-z .]+\(ANTARA\)\s*-\s*", " ", t)
    t = re.sub(r"^\s*KOMPAS\.com\s*-\s*", " ", t, flags=re.I)
    t = re.sub(r"\([^)]*\bANTARA\b[^)]*\)", " ", t)              # (ANTARA), (ANTARA/Nama), (ANTARA FOTO/...)
    t = re.sub(r"\bANTARA\b( FOTO)?", " ", t)                       # huruf besar saja, supaya kata biasa "antara" tidak ikut terhapus
    t = re.sub(r"\b(KOMPAS\.com|kompastekno|detikcom|detikNews|detikFinance|detikHealth|detikInet|detikHot)\b", " ", t, flags=re.I)
    t = re.sub(r"scroll to continue with content|baca juga|simak video", " ", t, flags=re.I)   # sisa tampilan situs web (iklan, tautan)
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def bersihkan(X):
    """Versi daftar-teks dari clean_text (dipakai sebagai langkah pertama di dalam Pipeline)."""
    return [clean_text(x) for x in X]

def clean_data(raw):
    """Buang baris yang tidak valid. Hasil: (tabel bersih, laporan jumlah baris yang dibuang per aturan)."""
    d = raw.copy()
    d["src"] = d.url.map(ambil_sumber)
    laporan = []

    def aturan(nama, keep):
        nonlocal d
        n0 = len(d)
        d = d[keep(d)]
        laporan.append((nama, n0 - len(d)))

    aturan("narasi kosong", lambda x: x.narasi.notna())
    aturan("label bukan 0/1", lambda x: x.label.isin([0, 1]))
    d = d.assign(text=d.narasi.map(clean_text))
    aturan(f"teks kurang dari {MIN_KATA} kata setelah dibersihkan", lambda x: x.text.str.split().str.len() >= MIN_KATA)
    aturan("duplikat (teks bersih sama persis)", lambda x: ~x.text.duplicated())
    d = d.assign(n_kata=d.text.str.split().str.len())
    lap = pd.DataFrame(laporan, columns=["aturan", "baris_dibuang"]).set_index("aturan")
    return d.reset_index(drop=True), lap

def prediksi(model, teks, batas_hoaks=0.65, batas_aman=0.35):
    """Kembalikan (zona, probabilitas_hoaks). Teks mentah langsung dimasukkan ke model."""
    if len(clean_text(teks).split()) < MIN_KATA:
        return "TERLALU PENDEK", None
    p = float(model.predict_proba([teks])[0][1])
    zona = "HOAKS" if p > batas_hoaks else "NON-HOAKS" if p < batas_aman else "TIDAK YAKIN"
    return zona, p
