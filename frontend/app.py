import streamlit as st
import requests
from PIL import Image
import csv
import os
from datetime import datetime

API_URL = "http://localhost:8000/predict"
FEEDBACK_FILE = "feedback.csv"

def save_feedback(predicted_label, confidence, correct_label):
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "predicted", "confidence", "correct_label"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            predicted_label,
            confidence,
            correct_label,
        ])

BAC_COLORS = {
    "Bac jaune": "#f5c518",
    "Colonne à verre": "#2e7d32",
    "Bac gris (ordures ménagères)": "#616161",
    "Bac marron (compost)": "#6d4c41",
}

CATEGORIES = [
    {
        "label": "Plastique",
        "bac": "Bac jaune",
        "color": "#f5c518",
        "exemples": "Bouteilles, flacons, barquettes, films plastique",
        "consigne": "Videz et rincez avant de déposer dans le bac jaune.",
        "emoji": "🧴",
    },
    {
        "label": "Carton",
        "bac": "Bac jaune",
        "color": "#f5c518",
        "exemples": "Boîtes, colis, emballages cartonnés",
        "consigne": "Aplatissez les cartons avant de les déposer.",
        "emoji": "📦",
    },
    {
        "label": "Papier",
        "bac": "Bac jaune",
        "color": "#f5c518",
        "exemples": "Journaux, magazines, feuilles, enveloppes",
        "consigne": "Pas de papier gras ni de mouchoirs usagés.",
        "emoji": "📄",
    },
    {
        "label": "Métal",
        "bac": "Bac jaune",
        "color": "#f5c518",
        "exemples": "Canettes, boîtes de conserve, capsules, aluminium",
        "consigne": "Écrasez les canettes pour gagner de la place.",
        "emoji": "🥫",
    },
    {
        "label": "Verre",
        "bac": "Colonne à verre",
        "color": "#2e7d32",
        "exemples": "Bouteilles en verre, bocaux, pots",
        "consigne": "Retirez les couvercles. Ne mettez pas les vitres ni la vaisselle.",
        "emoji": "🍾",
    },
    {
        "label": "Résidus",
        "bac": "Bac gris",
        "color": "#616161",
        "exemples": "Tout ce qui ne rentre pas dans les autres catégories",
        "consigne": "Déposez dans le bac gris. Ces déchets ne sont pas recyclables.",
        "emoji": "🗑️",
    },
]

HORS_PERIMETRE = [
    {"label": "Piles & batteries", "emoji": "🔋", "ou": "Bac à piles en magasin"},
    {"label": "Médicaments", "emoji": "💊", "ou": "Pharmacie"},
    {"label": "Textile & vêtements", "emoji": "👕", "ou": "Borne textile"},
    {"label": "Appareils électroniques", "emoji": "📱", "ou": "Déchetterie / DEEE"},
    {"label": "Huiles usagées", "emoji": "🛢️", "ou": "Déchetterie"},
    {"label": "Peintures & solvants", "emoji": "🎨", "ou": "Déchetterie"},
    {"label": "Ampoules", "emoji": "💡", "ou": "Point de collecte en magasin"},
]

st.set_page_config(page_title="Waste AI", page_icon="♻", layout="centered")

# Initialiser l'historique en session
if "historique" not in st.session_state:
    st.session_state.historique = []

tab1, tab2, tab3 = st.tabs(["Analyser un déchet", "Historique des scans", "Couverture & Limites"])

# ──────────────────────────────────────────
# TAB 1 — Analyse
# ──────────────────────────────────────────
with tab1:
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

                # Sauvegarder dans l'historique
                st.session_state.historique.append({
                    "heure": datetime.now().strftime("%H:%M:%S"),
                    "label": result["label"],
                    "bac": result["bac"],
                    "confidence": result["confidence"],
                })

                # Avertissement si confiance faible
                if result["confidence"] < 60:
                    st.warning(
                        "Confiance faible — l'objet est peut-être hors du périmètre du modèle "
                        "ou mal cadré. Consultez l'onglet **Couverture & Limites**."
                    )

                st.divider()
                st.markdown("**Ce résultat est incorrect ?**")
                with st.expander("Signaler une erreur"):
                    correct = st.selectbox(
                        "Quelle est la bonne catégorie ?",
                        ["Carton", "Verre", "Métal", "Papier", "Plastique", "Résidus", "Autre / Hors périmètre"],
                        key="correct_label",
                    )
                    if st.button("Envoyer le feedback", type="primary"):
                        save_feedback(result["label"], result["confidence"], correct)
                        st.success("Merci ! Votre correction a été enregistrée et servira à améliorer le modèle.")

            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter l'API. Vérifiez que le backend est lancé.")
            except Exception as e:
                st.error(f"Erreur : {e}")

# ──────────────────────────────────────────
# TAB 2 — Historique
# ──────────────────────────────────────────
with tab2:
    st.title("Historique des scans")
    st.caption("Tous les déchets analysés depuis le début de la session.")

    if not st.session_state.historique:
        st.info("Aucun scan effectué pour l'instant. Analysez un déchet dans l'onglet principal !")
    else:
        # Statistiques résumées
        total = len(st.session_state.historique)
        conf_moy = sum(s["confidence"] for s in st.session_state.historique) / total
        labels = [s["label"] for s in st.session_state.historique]
        plus_frequent = max(set(labels), key=labels.count)

        col1, col2, col3 = st.columns(3)
        col1.metric("Scans effectués", total)
        col2.metric("Confiance moyenne", f"{conf_moy:.1f}%")
        col3.metric("Déchet le plus scanné", plus_frequent)

        st.divider()

        # Tableau de l'historique
        for i, scan in enumerate(reversed(st.session_state.historique)):
            color = BAC_COLORS.get(scan["bac"], "#1976d2")
            conf = scan["confidence"]
            conf_color = "#2ecc71" if conf >= 80 else "#f39c12" if conf >= 60 else "#e74c3c"

            st.markdown(
                f"<div style='border:1px solid #333;border-radius:8px;padding:12px;margin-bottom:8px;"
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='color:#aaa;font-size:13px'>{scan['heure']}</span>"
                f"<span style='font-weight:bold;font-size:16px'>{scan['label']}</span>"
                f"<span style='background:{color};color:white;padding:4px 10px;"
                f"border-radius:12px;font-size:13px'>{scan['bac']}</span>"
                f"<span style='color:{conf_color};font-weight:bold'>{conf}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("Effacer l'historique", type="secondary"):
            st.session_state.historique = []
            st.rerun()

# ──────────────────────────────────────────
# TAB 3 — Couverture & Limites
# ──────────────────────────────────────────
with tab3:
    st.title("Couverture & Limites du modèle")
    st.caption("Ce que Waste AI sait reconnaître — et ce qu'il ne gère pas encore.")

    # Catégories couvertes
    st.subheader("Catégories couvertes (6)")
    st.info(
        "Le modèle a été entraîné sur **TrashNet** et **TACO**, deux datasets académiques "
        "de référence. Il reconnaît les 6 catégories ci-dessous, alignées sur les consignes ADEME / Citeo."
    )

    for cat in CATEGORIES:
        with st.expander(f"{cat['emoji']}  {cat['label']} — {cat['bac']}"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    f"<div style='background:{cat['color']};color:white;padding:12px;"
                    f"border-radius:8px;text-align:center;font-weight:bold'>{cat['bac']}</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(f"**Exemples :** {cat['exemples']}")
                st.markdown(f"**Consigne :** {cat['consigne']}")

    st.divider()

    # Hors périmètre
    st.subheader("Hors périmètre — non pris en charge")
    st.warning(
        "Ces catégories ne sont **pas reconnues** par le modèle actuel. "
        "Si vous photographiez l'un de ces objets, le résultat sera incorrect."
    )

    cols = st.columns(2)
    for i, item in enumerate(HORS_PERIMETRE):
        with cols[i % 2]:
            st.markdown(
                f"<div style='border:1px solid #444;border-radius:8px;padding:12px;margin-bottom:8px'>"
                f"<b>{item['emoji']} {item['label']}</b><br>"
                f"<span style='color:#aaa;font-size:13px'>→ {item['ou']}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Limites connues
    st.subheader("Limites connues du modèle")

    st.markdown("""
    | Situation | Impact | Recommandation |
    |-----------|--------|----------------|
    | Objet sur fond complexe | Précision réduite | Photographier sur fond neutre |
    | Plusieurs déchets dans l'image | Classification incorrecte | Un seul objet par photo |
    | Mauvaise luminosité | Précision réduite | Bonne lumière naturelle |
    | Objet abîmé ou écrasé | Peut être mal classé | Cadrer la partie la plus reconnaissable |
    | Score de confiance < 60% | Résultat peu fiable | Vérifier manuellement |
    """)

    st.divider()

    # Données d'entraînement
    st.subheader("Données d'entraînement")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset principal", "TrashNet")
        st.caption("~2 500 images sur fond blanc, 6 classes")
        st.metric("Dataset complémentaire", "TACO")
        st.caption("Photos en contexte réel, annotations manuelles")
    with col2:
        st.metric("Architecture", "MobileNetV3 Small")
        st.caption("Transfer learning — ImageNet → déchets")
        st.metric("Référentiel", "ADEME / Citeo")
        st.caption("Consignes de tri officielles France")
