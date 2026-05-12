import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000/predict"

BAC_COLORS = {
    "Bac jaune": "#f5c518",
    "Colonne à verre": "#2e7d32",
    "Bac gris (ordures ménagères)": "#616161",
    "Bac marron (compost)": "#6d4c41",
}

st.set_page_config(page_title="Waste AI", page_icon="♻", layout="centered")

st.title("Waste AI")
st.caption("Prenez en photo un déchet — l'IA vous dit où le jeter.")

source = st.radio("Source de l'image", ["Caméra", "Upload"], horizontal=True)

image_data = None

if source == "Caméra":
    image_data = st.camera_input("Pointez votre caméra vers le déchet")
else:
    image_data = st.file_uploader("Chargez une image", type=["jpg", "jpeg", "png", "webp"])

if image_data is not None:
    image = Image.open(image_data)
    st.image(image, caption="Image analysée", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        image_data.seek(0)
        try:
            response = requests.post(
                API_URL,
                files={"file": ("image.jpg", image_data, "image/jpeg")},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Déchet détecté", result["label"])
            with col2:
                st.metric("Confiance", f"{result['confidence']}%")

            bac = result["bac"]
            color = BAC_COLORS.get(bac, "#1976d2")
            st.markdown(
                f"<div style='background:{color};color:white;padding:16px;border-radius:8px;"
                f"font-size:18px;font-weight:bold;text-align:center'>{bac}</div>",
                unsafe_allow_html=True,
            )

            st.info(result["consigne"])

        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Vérifiez que le backend est lancé (`uvicorn main:app --reload`).")
        except Exception as e:
            st.error(f"Erreur : {e}")
