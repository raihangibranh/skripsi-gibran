# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# --- Load model ---
model = load_model('best_model_periodontal.keras')

# --- Fungsi prediksi ---
def predict(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    pred = model.predict(img_array)[0][0]
    if pred >= 0.5:
        return "Inflamasi"
    else:
        return "Normal"

# --- Streamlit UI ---
st.set_page_config(page_title="Klasifikasi Penyakit Periodontal", layout="wide")

# Judul utama
st.title("🦷 Klasifikasi Penyakit Periodontal")
st.write(
    "Unggah gambar jaringan gusi untuk mengetahui kondisinya. "
    "Aplikasi ini menggunakan model CNN berbasis MobileNetV2 "
    "untuk mendeteksi kondisi Normal atau Inflamasi."
)

# Layout dua kolom
col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Gambar Unggahan', use_container_width=True)

with col2:
    if uploaded_file is not None:
        with st.spinner("Memprediksi kondisi jaringan gusi..."):
            label = predict(img)

        st.subheader("Hasil Prediksi")

        # Ikon dan warna berbeda
        if label == "Inflamasi":
            st.error(f"❌ {label}")
        else:
            st.success(f"✅ {label}")

        # Penjelasan hasil
        st.markdown("**Interpretasi Hasil:**")
        if label == "Normal":
            st.write("Gusi sehat dan tidak ada tanda peradangan.")
            st.write("💡 Tips: Pertahankan kebiasaan menyikat gigi dengan benar dan rutin.")
        else:
            st.write("Gusi mengalami peradangan, kemungkinan risiko penyakit periodontal.")
            st.write("💡 Tips: Konsultasikan ke dokter gigi, perbaiki kebiasaan menyikat gigi, dan gunakan obat kumur antiseptik jika perlu.")

# Footer
st.markdown("---")
st.markdown("© 2025 Raihan Gibran Hidayat | Informatika")