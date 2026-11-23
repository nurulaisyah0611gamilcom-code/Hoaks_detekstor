"""Web app deteksi berita hoaks (Streamlit). Jalankan lokal: streamlit run app.py"""
import joblib
import streamlit as st
from teks import prediksi   # modul teks.py harus ada agar model bisa dimuat

st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="📰")

@st.cache_resource
def muat_model():
    return joblib.load("model.pkl")

model = muat_model()

st.title("📰 Deteksi Berita Hoaks Indonesia")
st.write("Tempel teks berita atau pesan berantai, lalu klik **Cek**.")
teks = st.text_area("Teks berita", height=220, placeholder="Tempel teks di sini...")

if st.button("Cek", type="primary"):
    zona, p = prediksi(model, teks)
    if zona == "TERLALU PENDEK":
        st.warning("Teks terlalu pendek. Masukkan minimal beberapa kalimat.")
    else:
        if zona == "HOAKS":
            st.error(f"Terindikasi HOAKS (probabilitas hoaks {p:.0%})")
        elif zona == "NON-HOAKS":
            st.success(f"Terindikasi NON-HOAKS (probabilitas hoaks {p:.0%})")
        else:
            st.warning(f"TIDAK YAKIN (probabilitas hoaks {p:.0%}). Cek manual ke sumber resmi.")
        st.progress(p, text=f"Probabilitas hoaks: {p:.0%}")

st.info(
    "Model ini menilai **gaya bahasa dan pola kata**, bukan kebenaran fakta. "
    "Berita asli yang pendek atau bergaya berbeda bisa salah dinilai. "
    "Selalu verifikasi ke sumber resmi atau turnbackhoax.id."
)
