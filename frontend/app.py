import streamlit as st
import requests
import math
from PIL import Image
import csv
import os
from datetime import datetime

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from streamlit_searchbox import st_searchbox
    SEARCHBOX_AVAILABLE = True
except ImportError:
    SEARCHBOX_AVAILABLE = False

API_URL = "http://localhost:8000/predict"
FEEDBACK_FILE = "feedback.csv"

st.set_page_config(
    page_title="Waste AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Design écoresponsable ──────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d1f14 0%, #1a2e1e 100%);
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="stTabs"] [role="tab"] {
    color: #95d5b2;
    font-weight: 600;
    font-size: 15px;
    padding: 8px 20px;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1b4332;
    color: #d8f3dc !important;
    border-bottom: 3px solid #52b788;
}

h1 { color: #d8f3dc !important; font-family: 'Georgia', serif; }
h2, h3 { color: #95d5b2 !important; }
p, label, .stMarkdown { color: #b7e4c7 !important; }

[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2d6a4f, #52b788) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid #52b788 !important;
    color: #95d5b2 !important;
    border-radius: 10px !important;
}

[data-testid="metric-container"] {
    background: #1b4332;
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #95d5b2 !important; }
[data-testid="stMetricValue"] { color: #d8f3dc !important; }

[data-testid="stInfo"] {
    background: #1b4332 !important;
    border-left: 4px solid #52b788 !important;
    color: #d8f3dc !important;
}
[data-testid="stWarning"] {
    background: #3d2b00 !important;
    border-left: 4px solid #f4a261 !important;
}
[data-testid="stSuccess"] {
    background: #1b4332 !important;
    border-left: 4px solid #52b788 !important;
}
[data-testid="stError"] {
    background: #3d0000 !important;
    border-left: 4px solid #e63946 !important;
}

[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: #1b4332 !important;
    border: 2px dashed #52b788 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] {
    background: #1b4332 !important;
    border: 1px solid #2d6a4f !important;
    border-radius: 10px !important;
}

hr { border-color: #2d6a4f !important; }
[data-testid="stRadio"] label { color: #b7e4c7 !important; }

/* Carte Folium — coins arrondis */
iframe {
    border-radius: 14px !important;
    overflow: hidden;
}

/* Searchbox adresse — style sombre */
[data-testid="stCustomComponentV1"] > div {
    border-radius: 12px;
}
div[class*="searchbox"] input,
div[class*="stSearchbox"] input {
    background: #1b4332 !important;
    color: #d8f3dc !important;
    border: 1px solid #2d6a4f !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers géolocalisation ────────────────────────────────────────────────────

def search_fr_address(query: str) -> list:
    """Autocomplete adresse via API Adresse data.gouv.fr — retourne (label, {lat, lon, label})."""
    if len(query) < 3:
        return []
    try:
        r = requests.get(
            "https://api-adresse.data.gouv.fr/search/",
            params={"q": query, "limit": 6},
            timeout=5,
        )
        r.raise_for_status()
        results = []
        for f in r.json().get("features", []):
            label = f["properties"]["label"]
            coords = f["geometry"]["coordinates"]
            results.append((label, {"lat": coords[1], "lon": coords[0], "label": label}))
        return results
    except Exception:
        return []


def geocode_address(address):
    """Fallback geocoding si l'autocomplete n'est pas utilisé."""
    try:
        r = requests.get(
            "https://api-adresse.data.gouv.fr/search/",
            params={"q": address, "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            return None, None, None
        coords = features[0]["geometry"]["coordinates"]
        label = features[0]["properties"]["label"]
        return coords[1], coords[0], label
    except Exception:
        return None, None, None


_OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def search_overpass(lat, lon, config):
    """Cherche des points de collecte via Overpass API (OpenStreetMap).
    Retourne (elements, error_message)."""
    radius = config["radius"]
    parts = []
    for q_template in config["queries"]:
        parts.append(q_template.format(radius=radius, lat=lat, lon=lon))
    query = f"[out:json][timeout:30];({''.join(parts)});out center;"

    for endpoint in _OVERPASS_ENDPOINTS:
        try:
            r = requests.get(endpoint, params={"data": query}, timeout=35)
            if r.status_code == 200:
                return r.json().get("elements", []), None
        except Exception:
            continue
    return [], "Impossible de contacter OpenStreetMap. Réessayez dans quelques secondes."


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en km entre deux coordonnées GPS."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Configuration "Où jeter ?" ─────────────────────────────────────────────────

OU_JETER_TYPES = {
    "🔋 Piles & Batteries": {
        "queries": [
            # Bornes de collecte dédiées
            'node["amenity"="recycling"]["recycling:batteries"="yes"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling:batteries"="yes"](around:{radius},{lat},{lon});',
            # Supermarchés & magasins qui ont une borne (quand taguée sur OSM)
            'node["shop"]["recycling:batteries"="yes"](around:{radius},{lat},{lon});',
            'way["shop"]["recycling:batteries"="yes"](around:{radius},{lat},{lon});',
        ],
        "radius": 5000,
        "conseil": "Bacs de collecte en supermarché, bureau de tabac, magasin d'électronique.",
        "eco_org": "Corepile / Screlec",
    },
    "📱 Électronique & DEEE": {
        "queries": [
            'node["amenity"="recycling"]["recycling:electronics"="yes"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling:electronics"="yes"](around:{radius},{lat},{lon});',
            # Déchetteries françaises : recycling_type=centre
            'node["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
        ],
        "radius": 10000,
        "conseil": "Le vendeur est obligé de reprendre votre ancien appareil lors d'un achat (loi AGEC).",
        "eco_org": "Ecosystem / Ecologic",
    },
    "👕 Vêtements & Textiles": {
        "queries": [
            'node["amenity"="recycling"]["recycling:clothes"="yes"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling:clothes"="yes"](around:{radius},{lat},{lon});',
        ],
        "radius": 5000,
        "conseil": "Bornes de collecte en ville, parking de supermarché.",
        "eco_org": "Le Relais / Emmaüs",
    },
    "💊 Médicaments": {
        "queries": [
            'node["amenity"="pharmacy"](around:{radius},{lat},{lon});',
            'way["amenity"="pharmacy"](around:{radius},{lat},{lon});',
        ],
        "radius": 3000,
        "conseil": "Rapportez vos médicaments non utilisés ou périmés en pharmacie (réseau Cyclamed).",
        "eco_org": "Cyclamed",
    },
    "💡 Ampoules": {
        "queries": [
            'node["amenity"="recycling"]["recycling:light_bulbs"="yes"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling:light_bulbs"="yes"](around:{radius},{lat},{lon});',
            # Centres de recyclage polyvalents
            'node["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
        ],
        "radius": 8000,
        "conseil": "Points de collecte en magasin de bricolage (Leroy Merlin, Ikea...) ou déchetterie.",
        "eco_org": "Ecosystem",
    },
    "🛋️ Encombrants & Déchetterie": {
        "queries": [
            # En France, les déchetteries sont taggées recycling_type=centre sur OSM
            'node["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            # Fallback avec waste_transfer_station (moins courant en France)
            'node["amenity"="waste_transfer_station"](around:{radius},{lat},{lon});',
            'way["amenity"="waste_transfer_station"](around:{radius},{lat},{lon});',
        ],
        "radius": 15000,
        "conseil": "Déchetterie pour meubles, matelas, électroménager volumineux, gravats.",
        "eco_org": "Mairie / Collectivité",
    },
    "🎨 Peintures & Solvants": {
        "queries": [
            'node["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'node["amenity"="waste_transfer_station"](around:{radius},{lat},{lon});',
            'way["amenity"="waste_transfer_station"](around:{radius},{lat},{lon});',
        ],
        "radius": 15000,
        "conseil": "Déchetterie uniquement. Ne jamais jeter dans les égouts ni à la poubelle.",
        "eco_org": "Déchetterie locale",
    },
    "🛢️ Huiles usagées": {
        "queries": [
            'node["amenity"="recycling"]["recycling:engine_oil"="yes"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling:engine_oil"="yes"](around:{radius},{lat},{lon});',
            'node["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
        ],
        "radius": 15000,
        "conseil": "Garages agréés et déchetteries. Ne jamais jeter dans les égouts.",
        "eco_org": "Huile de Vidange Pro",
    },
}


# ── Données statiques ──────────────────────────────────────────────────────────

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
    "Bac jaune":                              "#f5c518",
    "Colonne à verre":                        "#2d6a4f",
    "Bac gris (ordures ménagères)":           "#616161",
    "Bac marron (compost)":                   "#6d4c41",
    "Bac à piles (magasin / bureau de tabac)":"#e63946",
    "Point de collecte (magasin bricolage / Ikea)": "#f4a261",
    "Déchetterie (point DEEE)":               "#457b9d",
    "Pharmacie (réseau Cyclamed)":            "#7b2d8b",
}

CATEGORIES = [
    {
        "label": "Plastique", "bac": "Bac jaune", "color": "#f5c518",
        "exemples": "Bouteilles, flacons, barquettes, films plastique",
        "consigne": "Videz et rincez avant de déposer dans le bac jaune.", "emoji": "🧴",
    },
    {
        "label": "Carton", "bac": "Bac jaune", "color": "#f5c518",
        "exemples": "Boîtes, colis, emballages cartonnés",
        "consigne": "Aplatissez les cartons avant de les déposer.", "emoji": "📦",
    },
    {
        "label": "Papier", "bac": "Bac jaune", "color": "#f5c518",
        "exemples": "Journaux, magazines, feuilles, enveloppes",
        "consigne": "Pas de papier gras ni de mouchoirs usagés.", "emoji": "📄",
    },
    {
        "label": "Métal", "bac": "Bac jaune", "color": "#f5c518",
        "exemples": "Canettes, boîtes de conserve, capsules, aluminium",
        "consigne": "Écrasez les canettes pour gagner de la place.", "emoji": "🥫",
    },
    {
        "label": "Verre", "bac": "Colonne à verre", "color": "#2d6a4f",
        "exemples": "Bouteilles en verre, bocaux, pots",
        "consigne": "Retirez les couvercles. Ne mettez pas les vitres ni la vaisselle.", "emoji": "🍾",
    },
    {
        "label": "Résidus", "bac": "Bac gris", "color": "#616161",
        "exemples": "Tout ce qui ne rentre pas dans les autres catégories",
        "consigne": "Déposez dans le bac gris. Ces déchets ne sont pas recyclables.", "emoji": "🗑️",
    },
]

HORS_PERIMETRE = [
    {"label": "Textile & vêtements", "emoji": "👕", "ou": "Borne textile"},
    {"label": "Huiles usagées",      "emoji": "🛢️", "ou": "Déchetterie"},
    {"label": "Peintures & solvants","emoji": "🎨", "ou": "Déchetterie"},
]

ENCOMBRANTS = [
    {
        "label": "Meubles",
        "emoji": "🪑",
        "exemples": "Canapé, table, chaises, armoire, bureau",
        "conseil": "Déposez en déchetterie ou demandez l'enlèvement en mairie.",
    },
    {
        "label": "Matelas & literie",
        "emoji": "🛏️",
        "exemples": "Matelas, sommier, canapé-lit",
        "conseil": "Signalement possible via eco-mobilier.fr ou mairie.",
    },
    {
        "label": "Électroménager",
        "emoji": "🫙",
        "exemples": "Réfrigérateur, lave-linge, four, télévision",
        "conseil": "Reprise obligatoire par le vendeur lors d'un achat (loi AGEC). Sinon déchetterie.",
    },
    {
        "label": "Cartons volumineux",
        "emoji": "📦",
        "exemples": "Gros emballages de déménagement, palettes",
        "conseil": "Certaines mairies organisent une collecte dédiée. Sinon déchetterie bac carton.",
    },
    {
        "label": "Bricolage & jardinage",
        "emoji": "🪚",
        "exemples": "Bois, planches, tuyaux, outils, terreau, branchages",
        "conseil": "Déchetterie uniquement. Les branchages peuvent aller en composterie.",
    },
    {
        "label": "Gravats",
        "emoji": "🧱",
        "exemples": "Parpaings, carrelage, plâtre, béton",
        "conseil": "Déchetterie zone gravats (volume souvent limité). Sinon benne privée.",
    },
]

if "historique" not in st.session_state:
    st.session_state.historique = []

# ── En-tête ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:24px 0 8px 0'>
    <span style='font-size:52px'>♻️</span>
    <h1 style='margin:4px 0; font-size:2.4rem; letter-spacing:1px; color:#d8f3dc'>Waste AI</h1>
    <p style='color:#95d5b2; margin:4px 0 0 0; font-size:15px'>
        Photographiez un déchet — l'IA vous dit où le jeter.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Analyser", "🗺️ Où jeter ?", "🛋️ Encombrants", "📋 Historique", "ℹ️ Couverture & Limites"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Analyse
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Upload"], horizontal=True)

    image_data = None
    if source == "📷 Caméra":
        image_data = st.camera_input("Pointez votre caméra vers le déchet")
    else:
        image_data = st.file_uploader("Chargez une image", type=["jpg", "jpeg", "png", "webp"])

    if image_data is not None:
        image = Image.open(image_data)
        st.image(image, caption="Image analysée", use_container_width=True)

        with st.spinner("🌿 Analyse en cours..."):
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
                color = BAC_COLORS.get(bac, "#2d6a4f")
                st.markdown(
                    f"<div style='background:{color};color:white;padding:18px;"
                    f"border-radius:12px;font-size:20px;font-weight:bold;"
                    f"text-align:center;margin:12px 0;letter-spacing:0.5px'>"
                    f"🗑️ {bac}</div>",
                    unsafe_allow_html=True,
                )
                st.info(f"💡 {result['consigne']}")

                st.session_state.historique.append({
                    "heure": datetime.now().strftime("%H:%M:%S"),
                    "label": result["label"],
                    "bac": result["bac"],
                    "confidence": result["confidence"],
                })

                if result["confidence"] < 60:
                    st.warning(
                        "⚠️ Confiance faible — l'objet est peut-être hors du périmètre "
                        "ou mal cadré. Consultez l'onglet **Couverture & Limites**."
                    )

                st.divider()
                st.markdown("**Ce résultat est incorrect ?**")
                with st.expander("Signaler une erreur"):
                    correct = st.selectbox(
                        "Quelle est la bonne catégorie ?",
                        ["Carton", "Verre", "Métal", "Papier", "Plastique",
                         "Résidus", "Autre / Hors périmètre"],
                        key="correct_label",
                    )
                    if st.button("Envoyer le feedback", type="primary"):
                        save_feedback(result["label"], result["confidence"], correct)
                        st.success("Merci ! Votre correction a été enregistrée.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de contacter l'API. Vérifiez que le backend est lancé.")
            except Exception as e:
                st.error(f"Erreur : {e}")
    else:
        st.markdown("""
        <div style='text-align:center; padding:40px 20px; border:2px dashed #2d6a4f;
                    border-radius:16px; margin-top:20px'>
            <div style='font-size:48px'>📸</div>
            <p style='color:#95d5b2; margin:8px 0 0 0'>
                Prenez une photo ou uploadez une image pour commencer
            </p>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Où jeter ?
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("🗺️ Où jeter ?")
    st.caption(
        "Pour les déchets complexes — piles, médicaments, vêtements, électronique... "
        "Sélectionnez le type et entrez votre adresse."
    )

    type_dechet = st.selectbox(
        "Type de déchet",
        list(OU_JETER_TYPES.keys()),
        key="ou_jeter_type",
    )

    config = OU_JETER_TYPES[type_dechet]

    st.markdown(
        f"<div style='background:#1b4332; border-left:4px solid #52b788; border-radius:8px;"
        f"padding:12px 16px; margin:8px 0 16px 0'>"
        f"<b style='color:#95d5b2'>💡 Conseil :</b> "
        f"<span style='color:#b7e4c7'>{config['conseil']}</span><br>"
        f"<b style='color:#95d5b2'>🏢 Éco-organisme :</b> "
        f"<span style='color:#b7e4c7'>{config['eco_org']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Saisie adresse avec autocomplete
    st.markdown(
        "<p style='color:#95d5b2; font-weight:600; margin:16px 0 4px 0; font-size:15px'>"
        "📍 Votre adresse</p>",
        unsafe_allow_html=True,
    )
    if SEARCHBOX_AVAILABLE:
        selected_addr = st_searchbox(
            search_fr_address,
            placeholder="Ex: 10 rue de Rivoli, Paris",
            label=None,
            key="ou_jeter_searchbox",
            clear_on_submit=False,
        )
    else:
        st.caption("Autocomplete non disponible — saisie manuelle")
        raw_addr = st.text_input(
            "",
            placeholder="Ex: 10 rue de Rivoli, Paris",
            key="ou_jeter_adresse",
            label_visibility="collapsed",
        )
        selected_addr = None

    rechercher = st.button("🔍 Rechercher les points de collecte", type="primary", key="ou_jeter_btn")

    if rechercher:
        lat = lon = label_addr = None

        if SEARCHBOX_AVAILABLE:
            if isinstance(selected_addr, dict):
                lat = selected_addr["lat"]
                lon = selected_addr["lon"]
                label_addr = selected_addr["label"]
            else:
                st.warning("Sélectionnez une adresse dans la liste déroulante qui s'affiche au fil de la frappe.")
        else:
            if raw_addr.strip():
                with st.spinner("📍 Géolocalisation en cours..."):
                    lat, lon, label_addr = geocode_address(raw_addr)
                if lat is None:
                    st.error("❌ Adresse introuvable. Essayez avec le nom de ville ou code postal.")
            else:
                st.warning("Veuillez entrer une adresse.")

        if lat is not None:
            st.success(f"📍 **{label_addr}**")

            with st.spinner("🗺️ Recherche des points de collecte..."):
                elements, err = search_overpass(lat, lon, config)

            if err:
                st.error(f"❌ {err}")
            else:
                # Filtre et tri par distance
                results = []
                seen = set()
                for el in elements:
                    el_lat = el.get("lat") or (el.get("center") or {}).get("lat")
                    el_lon = el.get("lon") or (el.get("center") or {}).get("lon")
                    if not (el_lat and el_lon):
                        continue
                    key = (round(el_lat, 5), round(el_lon, 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    el["_lat"] = el_lat
                    el["_lon"] = el_lon
                    el["_distance"] = haversine_km(lat, lon, el_lat, el_lon)
                    results.append(el)

                results.sort(key=lambda x: x["_distance"])
                top = results[:15]

                if not top:
                    st.warning(
                        f"😕 Aucun point trouvé dans un rayon de {config['radius'] // 1000} km. "
                        "Essayez une adresse dans une ville plus grande, ou contactez votre mairie."
                    )
                else:
                    st.success(
                        f"✅ {len(results)} point(s) trouvé(s) "
                        f"dans un rayon de {config['radius'] // 1000} km"
                    )

                    # Carte interactive
                    if FOLIUM_AVAILABLE:
                        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
                        m.get_root().html.add_child(folium.Element(
                            "<style>.leaflet-control-attribution { display: none !important; }</style>"
                        ))
                        folium.Marker(
                            [lat, lon],
                            popup="📍 Votre position",
                            icon=folium.Icon(color="red", icon="home"),
                        ).add_to(m)
                        for el in top:
                            tags = el.get("tags", {})
                            name = tags.get("name") or tags.get("operator") or "Point de collecte"
                            addr_parts = [
                                tags.get("addr:housenumber", ""),
                                tags.get("addr:street", ""),
                                tags.get("addr:city", ""),
                            ]
                            addr_str = " ".join(p for p in addr_parts if p)
                            popup_html = (
                                f"<b>{name}</b>"
                                + (f"<br>{addr_str}" if addr_str else "")
                                + f"<br><b>📏 {el['_distance']:.1f} km</b>"
                            )
                            folium.Marker(
                                [el["_lat"], el["_lon"]],
                                popup=folium.Popup(popup_html, max_width=220),
                                icon=folium.Icon(color="green", icon="leaf"),
                            ).add_to(m)
                        st_folium(m, width=None, height=420, returned_objects=[])
                    else:
                        st.info("Pour la carte : `pip install folium streamlit-folium`")

                    # Liste des résultats
                    st.subheader(f"Les {min(10, len(top))} points les plus proches")
                    for i, el in enumerate(top[:10], 1):
                        tags = el.get("tags", {})
                        name = tags.get("name") or tags.get("operator") or "Point de collecte"
                        addr_num = tags.get("addr:housenumber", "")
                        addr_street = tags.get("addr:street", "")
                        addr_city = tags.get("addr:city", "")
                        addr_full = " ".join(p for p in [addr_num, addr_street] if p)
                        if addr_city:
                            addr_full = f"{addr_full}, {addr_city}" if addr_full else addr_city
                        opening = tags.get("opening_hours", "")

                        html = (
                            f"<div style='background:#1b4332; border:1px solid #2d6a4f;"
                            f"border-radius:10px; padding:14px; margin-bottom:8px'>"
                            f"<div style='display:flex; justify-content:space-between; align-items:center'>"
                            f"<b style='color:#d8f3dc; font-size:16px'>{i}. {name}</b>"
                            f"<span style='background:#2d6a4f; color:#d8f3dc; padding:4px 12px;"
                            f"border-radius:20px; font-size:13px'>📏 {el['_distance']:.1f} km</span>"
                            f"</div>"
                        )
                        if addr_full:
                            html += f"<p style='color:#95d5b2; font-size:13px; margin:6px 0 0 0'>📍 {addr_full}</p>"
                        if opening:
                            html += f"<p style='color:#95d5b2; font-size:13px; margin:4px 0 0 0'>🕐 {opening}</p>"
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Encombrants
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("🛋️ Gros objets & Encombrants")
    st.caption("Meubles, matelas, électroménager — que faire quand c'est trop grand pour la poubelle ?")

    st.markdown("""
    <div style='background:#1b4332; border-left:4px solid #52b788; border-radius:8px;
                padding:16px 20px; margin-bottom:16px'>
        <b style='color:#d8f3dc; font-size:16px'>🏛️ Service municipal d'enlèvement des encombrants</b>
        <p style='color:#b7e4c7; margin:12px 0 0 0; line-height:1.7'>
            Dans la plupart des communes françaises, la mairie propose un service gratuit de collecte
            des encombrants à domicile. Voici comment ça fonctionne :<br><br>
            <b style='color:#95d5b2'>① Contactez votre mairie</b> (par téléphone ou sur leur site)<br>
            <b style='color:#95d5b2'>② Déclarez votre encombrant</b> (type, taille approximative)<br>
            <b style='color:#95d5b2'>③ Recevez un numéro de passage</b> à scotcher sur l'objet<br>
            <b style='color:#95d5b2'>④ Déposez l'objet sur le trottoir</b> la veille du jour indiqué<br>
            <b style='color:#95d5b2'>⑤ Les agents passent</b> et récupèrent l'objet
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:12px; padding:16px'>
            <b style='color:#d8f3dc'>📞 Trouver votre mairie</b><br>
            <p style='color:#95d5b2; font-size:13px; margin:8px 0 0 0'>
            Recherchez <i>"enlèvement encombrants + [votre commune]"</i>
            ou appelez directement votre mairie.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:12px; padding:16px'>
            <b style='color:#d8f3dc'>🗺️ Trouver une déchetterie</b><br>
            <p style='color:#95d5b2; font-size:13px; margin:8px 0 0 0'>
            Utilisez l'onglet <b>Où jeter ?</b> → Encombrants & Déchetterie
            pour trouver la déchetterie la plus proche.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Que faire selon le type d'objet ?")

    for item in ENCOMBRANTS:
        with st.expander(f"{item['emoji']}  {item['label']}"):
            st.markdown(f"**Exemples :** {item['exemples']}")
            st.markdown(
                f"<div style='background:#2d6a4f; border-radius:8px; padding:10px; color:#d8f3dc'>"
                f"✅ {item['conseil']}</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("♻️ Avant de jeter — pensez au réemploi !")
    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px'>
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:10px; padding:14px'>
            <b style='color:#95d5b2'>🛒 Le Bon Coin</b><br>
            <span style='color:#b7e4c7; font-size:13px'>Donnez ou vendez vos meubles</span>
        </div>
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:10px; padding:14px'>
            <b style='color:#95d5b2'>🤝 Emmaüs / La Croix-Rouge</b><br>
            <span style='color:#b7e4c7; font-size:13px'>Dons de mobilier en bon état</span>
        </div>
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:10px; padding:14px'>
            <b style='color:#95d5b2'>📦 Geev</b><br>
            <span style='color:#b7e4c7; font-size:13px'>Application de dons entre particuliers</span>
        </div>
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:10px; padding:14px'>
            <b style='color:#95d5b2'>🌿 Ecosystem</b><br>
            <span style='color:#b7e4c7; font-size:13px'>Reprise de l'électroménager et des ampoules</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Historique
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.title("📋 Historique des scans")
    st.caption("Tous les déchets analysés depuis le début de la session.")

    if not st.session_state.historique:
        st.markdown("""
        <div style='text-align:center; padding:40px; border:2px dashed #2d6a4f;
                    border-radius:16px; margin-top:20px'>
            <div style='font-size:40px'>🌿</div>
            <p style='color:#95d5b2'>Aucun scan effectué. Analysez un déchet dans l'onglet Analyser !</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(st.session_state.historique)
        conf_moy = sum(s["confidence"] for s in st.session_state.historique) / total
        labels = [s["label"] for s in st.session_state.historique]
        plus_frequent = max(set(labels), key=labels.count)

        col1, col2, col3 = st.columns(3)
        col1.metric("Scans effectués", total)
        col2.metric("Confiance moyenne", f"{conf_moy:.1f}%")
        col3.metric("Déchet le + scanné", plus_frequent)

        st.divider()

        for scan in reversed(st.session_state.historique):
            color = BAC_COLORS.get(scan["bac"], "#2d6a4f")
            conf = scan["confidence"]
            conf_color = "#52b788" if conf >= 80 else "#f4a261" if conf >= 60 else "#e63946"

            st.markdown(
                f"<div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:10px;"
                f"padding:14px; margin-bottom:8px; display:flex; justify-content:space-between;"
                f"align-items:center'>"
                f"<span style='color:#95d5b2; font-size:13px'>🕐 {scan['heure']}</span>"
                f"<span style='color:#d8f3dc; font-weight:bold; font-size:16px'>{scan['label']}</span>"
                f"<span style='background:{color}; color:white; padding:4px 12px;"
                f"border-radius:20px; font-size:13px'>{scan['bac']}</span>"
                f"<span style='color:{conf_color}; font-weight:bold'>{conf}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("🗑️ Effacer l'historique", type="secondary"):
            st.session_state.historique = []
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Couverture & Limites
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.title("ℹ️ Couverture & Limites du modèle")
    st.caption("Ce que Waste AI sait reconnaître — et ce qu'il ne gère pas encore.")

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

    st.subheader("Hors périmètre — non pris en charge par l'IA")
    st.warning(
        "Ces catégories ne sont **pas reconnues** par le modèle. "
        "Utilisez l'onglet **Où jeter ?** pour trouver les points de collecte adaptés."
    )

    cols = st.columns(2)
    for i, item in enumerate(HORS_PERIMETRE):
        with cols[i % 2]:
            st.markdown(
                f"<div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:8px;"
                f"padding:12px; margin-bottom:8px'>"
                f"<b style='color:#d8f3dc'>{item['emoji']} {item['label']}</b><br>"
                f"<span style='color:#95d5b2; font-size:13px'>→ {item['ou']}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("Limites connues du modèle")
    st.markdown("""
    | Situation | Impact | Recommandation |
    |-----------|--------|----------------|
    | Objet sur fond complexe | Précision réduite | Photographier sur fond neutre |
    | Plusieurs déchets dans l'image | Classification incorrecte | Un seul objet par photo |
    | Mauvaise luminosité | Précision réduite | Bonne lumière naturelle |
    | Objet abîmé ou écrasé | Peut être mal classé | Cadrer la partie la plus reconnaissable |
    | Score de confiance < 60% | Résultat peu fiable | Vérifier manuellement |
    | Gros encombrant | Non reconnu par l'IA | Utiliser l'onglet Encombrants |
    """)

    st.divider()

    st.subheader("Données d'entraînement")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset principal", "TrashNet")
        st.caption("~2 500 images sur fond blanc, 6 classes")
        st.metric("Dataset complémentaire", "TACO")
        st.caption("Photos en contexte réel, annotations manuelles")
    with col2:
        st.metric("Architecture", "EfficientNet-B2")
        st.caption("Transfer learning — ImageNet → déchets")
        st.metric("Référentiel", "ADEME / Citeo")
        st.caption("Consignes de tri officielles France")
