import streamlit as st
import streamlit.components.v1 as components
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import math
import html as html_lib
import requests
import os
from datetime import datetime
from pathlib import Path

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from torchvision import models

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from supabase import create_client as _sb_create
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    from streamlit_searchbox import st_searchbox
    SEARCHBOX_AVAILABLE = True
except ImportError:
    SEARCHBOX_AVAILABLE = False

# ── Sécurité ───────────────────────────────────────────────────────────────────
Image.MAX_IMAGE_PIXELS = 50_000_000


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# ── Modèle ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "waste_ai_v4.pt"
NUM_CLASSES = 6
IMG_SIZE = 260

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CATEGORIES = {
    0: {"label": "Carton",    "bac": "Bac jaune",                    "consigne": "Aplatissez le carton et déposez-le dans le bac jaune."},
    1: {"label": "Verre",     "bac": "Colonne à verre",              "consigne": "Déposez dans une colonne à verre. Ne mettez pas le couvercle."},
    2: {"label": "Métal",     "bac": "Bac jaune",                    "consigne": "Déposez dans le bac jaune. Écrasez les canettes si possible."},
    3: {"label": "Papier",    "bac": "Bac jaune",                    "consigne": "Déposez dans le bac jaune. Pas de papier gras ni de mouchoirs."},
    4: {"label": "Plastique", "bac": "Bac jaune",                    "consigne": "Déposez dans le bac jaune. Videz et rincez les emballages."},
    5: {"label": "Résidus",   "bac": "Bac gris (ordures ménagères)", "consigne": "Déposez dans le bac gris. Cet objet n'est pas recyclable."},
}


@st.cache_resource(show_spinner="🌿 Chargement du modèle IA...")
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        if TIMM_AVAILABLE:
            m = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
        else:
            m = models.mobilenet_v3_small(weights=None)
            m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        return m
    except Exception:
        return None


def predict_image(image_data):
    model = load_model()
    if model is None:
        return None, "Modèle IA non disponible."

    raw = image_data.read()
    if len(raw) > MAX_FILE_SIZE_BYTES:
        return None, f"Image trop volumineuse (max 10 Mo)."

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        return None, "Fichier invalide — envoyez une image JPG ou PNG."

    tensor = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        proba = torch.softmax(output, dim=1)
        confidence, class_id = torch.max(proba, dim=1)

    cat = CATEGORIES[class_id.item()]
    return {
        "label": cat["label"],
        "bac": cat["bac"],
        "consigne": cat["consigne"],
        "confidence": round(confidence.item() * 100, 1),
    }, None


# ── Géolocalisation ────────────────────────────────────────────────────────────

def search_fr_address(query: str) -> list:
    if len(query) < 3 or len(query) > 200:
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


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


_OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def search_overpass(lat, lon, config):
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


# ── Config "Où jeter ?" ────────────────────────────────────────────────────────

OU_JETER_TYPES = {
    "🔋 Piles & Batteries": {
        "queries": [
            'node["amenity"="recycling"]["recycling:batteries"="yes"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling:batteries"="yes"](around:{radius},{lat},{lon});',
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
            'node["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
        ],
        "radius": 8000,
        "conseil": "Points de collecte en magasin de bricolage (Leroy Merlin, Ikea...) ou déchetterie.",
        "eco_org": "Ecosystem",
    },
    "🛋️ Encombrants & Déchetterie": {
        "queries": [
            'node["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
            'way["amenity"="recycling"]["recycling_type"="centre"](around:{radius},{lat},{lon});',
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

# ── Supabase — comptes & historique persistant ────────────────────────────────
_SB_URL = os.environ.get("SUPABASE_URL", "")
_SB_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
DB_READY = SUPABASE_AVAILABLE and bool(_SB_URL) and bool(_SB_KEY)


def _authed_client(access_token: str):
    """Client Supabase authentifié avec le JWT de l'utilisateur."""
    client = _sb_create(_SB_URL, _SB_KEY)
    client.postgrest.auth(access_token)
    return client


def auth_login(email: str, password: str):
    try:
        client = _sb_create(_SB_URL, _SB_KEY)
        r = client.auth.sign_in_with_password({"email": email, "password": password})
        return r.user, r.session, None
    except Exception as e:
        return None, None, "Email ou mot de passe incorrect."


def auth_signup(email: str, password: str):
    try:
        client = _sb_create(_SB_URL, _SB_KEY)
        r = client.auth.sign_up({"email": email, "password": password})
        return r.user, r.session, None
    except Exception as e:
        return None, None, str(e)


def db_save_scan(user_id: str, token: str, label: str, bac: str, confidence: float):
    try:
        _authed_client(token).table("scans").insert({
            "user_id": user_id, "label": label,
            "bac": bac, "confidence": confidence,
        }).execute()
    except Exception:
        pass


def db_get_scans(user_id: str, token: str) -> list:
    try:
        r = (_authed_client(token).table("scans")
             .select("label, bac, confidence, scanned_at")
             .eq("user_id", user_id)
             .order("scanned_at", desc=True)
             .limit(200)
             .execute())
        return r.data or []
    except Exception:
        return []


def db_clear_scans(user_id: str, token: str):
    try:
        _authed_client(token).table("scans").delete().eq("user_id", user_id).execute()
    except Exception:
        pass


def save_feedback(predicted_label, confidence, correct_label):
    if DB_READY:
        try:
            _sb_create(_SB_URL, _SB_KEY).table("feedback").insert({
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct_label": correct_label,
            }).execute()
        except Exception:
            pass


BAC_COLORS = {
    "Bac jaune":                                   "#f5c518",
    "Colonne à verre":                             "#2d6a4f",
    "Bac gris (ordures ménagères)":                "#616161",
    "Bac marron (compost)":                        "#6d4c41",
    "Bac à piles (magasin / bureau de tabac)":     "#e63946",
    "Point de collecte (magasin bricolage / Ikea)":"#f4a261",
    "Déchetterie (point DEEE)":                    "#457b9d",
    "Pharmacie (réseau Cyclamed)":                 "#7b2d8b",
}

CATEGORIES_DISPLAY = [
    {"label": "Plastique", "bac": "Bac jaune",        "color": "#f5c518", "emoji": "🧴",
     "exemples": "Bouteilles, flacons, barquettes, films plastique",
     "consigne": "Videz et rincez avant de déposer dans le bac jaune."},
    {"label": "Carton",    "bac": "Bac jaune",        "color": "#f5c518", "emoji": "📦",
     "exemples": "Boîtes, colis, emballages cartonnés",
     "consigne": "Aplatissez les cartons avant de les déposer."},
    {"label": "Papier",    "bac": "Bac jaune",        "color": "#f5c518", "emoji": "📄",
     "exemples": "Journaux, magazines, feuilles, enveloppes",
     "consigne": "Pas de papier gras ni de mouchoirs usagés."},
    {"label": "Métal",     "bac": "Bac jaune",        "color": "#f5c518", "emoji": "🥫",
     "exemples": "Canettes, boîtes de conserve, capsules, aluminium",
     "consigne": "Écrasez les canettes pour gagner de la place."},
    {"label": "Verre",     "bac": "Colonne à verre",  "color": "#2d6a4f", "emoji": "🍾",
     "exemples": "Bouteilles en verre, bocaux, pots",
     "consigne": "Retirez les couvercles. Ne mettez pas les vitres ni la vaisselle."},
    {"label": "Résidus",   "bac": "Bac gris",         "color": "#616161", "emoji": "🗑️",
     "exemples": "Tout ce qui ne rentre pas dans les autres catégories",
     "consigne": "Déposez dans le bac gris. Ces déchets ne sont pas recyclables."},
]

HORS_PERIMETRE = [
    {"label": "Textile & vêtements", "emoji": "👕", "ou": "Borne textile"},
    {"label": "Huiles usagées",       "emoji": "🛢️", "ou": "Déchetterie"},
    {"label": "Peintures & solvants", "emoji": "🎨", "ou": "Déchetterie"},
]

ENCOMBRANTS = [
    {"label": "Meubles",            "emoji": "🪑",
     "exemples": "Canapé, table, chaises, armoire, bureau",
     "conseil": "Déposez en déchetterie ou demandez l'enlèvement en mairie."},
    {"label": "Matelas & literie",  "emoji": "🛏️",
     "exemples": "Matelas, sommier, canapé-lit",
     "conseil": "Signalement possible via eco-mobilier.fr ou mairie."},
    {"label": "Électroménager",     "emoji": "🫙",
     "exemples": "Réfrigérateur, lave-linge, four, télévision",
     "conseil": "Reprise obligatoire par le vendeur lors d'un achat (loi AGEC). Sinon déchetterie."},
    {"label": "Cartons volumineux", "emoji": "📦",
     "exemples": "Gros emballages de déménagement, palettes",
     "conseil": "Certaines mairies organisent une collecte dédiée. Sinon déchetterie bac carton."},
    {"label": "Bricolage & jardinage", "emoji": "🪚",
     "exemples": "Bois, planches, tuyaux, outils, terreau, branchages",
     "conseil": "Déchetterie uniquement. Les branchages peuvent aller en composterie."},
    {"label": "Gravats",            "emoji": "🧱",
     "exemples": "Parpaings, carrelage, plâtre, béton",
     "conseil": "Déchetterie zone gravats (volume souvent limité). Sinon benne privée."},
]

# ── Streamlit config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Waste AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={},
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d1f14 0%, #1a2e1e 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stTabs"] [role="tab"] {
    color: #95d5b2; font-weight: 600; font-size: 15px;
    padding: 8px 20px; border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1b4332; color: #d8f3dc !important;
    border-bottom: 3px solid #52b788;
}

h1 { color: #d8f3dc !important; font-family: 'Georgia', serif; }
h2, h3 { color: #95d5b2 !important; }
p, label, .stMarkdown { color: #b7e4c7 !important; }

[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2d6a4f, #52b788) !important;
    border: none !important; color: white !important;
    font-weight: bold !important; border-radius: 10px !important;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important; border: 1px solid #52b788 !important;
    color: #95d5b2 !important; border-radius: 10px !important;
}
[data-testid="metric-container"] {
    background: #1b4332; border: 1px solid #2d6a4f;
    border-radius: 12px; padding: 16px;
}
[data-testid="stMetricLabel"] { color: #95d5b2 !important; }
[data-testid="stMetricValue"] { color: #d8f3dc !important; }
[data-testid="stInfo"] {
    background: #1b4332 !important; border-left: 4px solid #52b788 !important;
    color: #d8f3dc !important;
}
[data-testid="stWarning"] {
    background: #3d2b00 !important; border-left: 4px solid #f4a261 !important;
}
[data-testid="stSuccess"] {
    background: #1b4332 !important; border-left: 4px solid #52b788 !important;
}
[data-testid="stError"] {
    background: #3d0000 !important; border-left: 4px solid #e63946 !important;
}
[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: #1b4332 !important; border: 2px dashed #52b788 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] {
    background: #1b4332 !important; border: 1px solid #2d6a4f !important;
    border-radius: 10px !important;
}
hr { border-color: #2d6a4f !important; }
[data-testid="stRadio"] label { color: #b7e4c7 !important; }
iframe { border-radius: 14px !important; overflow: hidden; }

/* Fix streamlit-folium iframe white space */
[data-testid="stCustomComponentV1"] {
    height: 300px !important;
    min-height: unset !important;
    overflow: hidden !important;
}

@media (max-width: 768px) {
    [data-testid="stTabs"] [role="tab"] {
        font-size: 12px !important;
        padding: 6px 10px !important;
    }
    h1 { font-size: 1.6rem !important; }
    [data-testid="metric-container"] { padding: 10px !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    .block-container { padding: 1rem 0.75rem !important; }
}
</style>
""", unsafe_allow_html=True)


if "historique" not in st.session_state:
    st.session_state.historique = []
if "user" not in st.session_state:
    st.session_state.user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "landing_done" not in st.session_state:
    st.session_state.landing_done = False

# ── Navbar ─────────────────────────────────────────────────────────────────────
col_brand, col_fill, col_auth = st.columns([5, 1, 2])

with col_brand:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; padding:10px 0 6px 0'>
        <span style='font-size:36px; line-height:1'>♻️</span>
        <div>
            <div style='color:#d8f3dc; font-size:22px; font-weight:bold;
                        font-family:Georgia,serif; line-height:1.2'>Waste AI</div>
            <div style='color:#95d5b2; font-size:12px'>
                Photographiez un déchet — l'IA vous dit où le jeter.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_auth:
    if st.session_state.user:
        email_short = st.session_state.user["email"].split("@")[0]
        st.markdown(
            f"<div style='color:#95d5b2; font-size:12px; text-align:right;"
            f"padding-top:6px'>👤 {email_short}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Déconnexion", type="secondary", use_container_width=True):
            st.session_state.user = None
            st.session_state.access_token = None
            st.rerun()
    elif DB_READY:
        with st.popover("👤 Connexion", use_container_width=True):
            tab_in, tab_up = st.tabs(["Se connecter", "Créer un compte"])

            with tab_in:
                email_in = st.text_input("Email", key="login_email")
                pw_in = st.text_input("Mot de passe", type="password", key="login_pw")
                if st.button("Se connecter", type="primary", key="login_btn", use_container_width=True):
                    if email_in and pw_in:
                        user, session, err = auth_login(email_in, pw_in)
                        if user:
                            st.session_state.user = {"id": str(user.id), "email": user.email}
                            st.session_state.access_token = session.access_token
                            st.rerun()
                        else:
                            st.error(err)
                    else:
                        st.warning("Remplissez tous les champs.")

            with tab_up:
                email_up = st.text_input("Email", key="signup_email")
                pw_up = st.text_input("Mot de passe", type="password", key="signup_pw")
                pw_up2 = st.text_input("Confirmer", type="password", key="signup_pw2")
                if st.button("Créer mon compte", type="primary", key="signup_btn", use_container_width=True):
                    if not (email_up and pw_up and pw_up2):
                        st.warning("Remplissez tous les champs.")
                    elif pw_up != pw_up2:
                        st.error("Les mots de passe ne correspondent pas.")
                    elif len(pw_up) < 6:
                        st.error("Mot de passe trop court (6 caractères min).")
                    else:
                        user, session, err = auth_signup(email_up, pw_up)
                        if user:
                            if session:
                                st.session_state.user = {"id": str(user.id), "email": user.email}
                                st.session_state.access_token = session.access_token
                                st.rerun()
                            else:
                                st.success("✅ Compte créé ! Connectez-vous ci-dessus.")
                        else:
                            st.error(f"Erreur : {err}")

st.markdown("<hr style='border-color:#1b4332; margin:4px 0 12px 0'>", unsafe_allow_html=True)

# ── Landing page (première visite) ────────────────────────────────────────────
if not st.session_state.landing_done:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0'>
        <h2 style='color:#d8f3dc; font-size:1.5rem; margin-bottom:4px'>
            Bienvenue sur Waste AI ♻️
        </h2>
        <p style='color:#95d5b2; font-size:15px; max-width:480px; margin:0 auto 20px auto; line-height:1.6'>
            Prenez en photo n'importe quel déchet — l'IA identifie sa catégorie
            et vous dit exactement où le jeter, près de chez vous.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:12px; padding:16px; text-align:center'>
            <div style='font-size:28px'>📸</div>
            <b style='color:#d8f3dc'>Photographiez</b>
            <p style='color:#95d5b2; font-size:13px; margin:6px 0 0 0'>Caméra ou upload — un seul objet par photo</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:12px; padding:16px; text-align:center'>
            <div style='font-size:28px'>🤖</div>
            <b style='color:#d8f3dc'>L'IA analyse</b>
            <p style='color:#95d5b2; font-size:13px; margin:6px 0 0 0'>EfficientNet-B2 — 6 catégories de déchets</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:12px; padding:16px; text-align:center'>
            <div style='font-size:28px'>🗺️</div>
            <b style='color:#d8f3dc'>Trouvez où jeter</b>
            <p style='color:#95d5b2; font-size:13px; margin:6px 0 0 0'>Points de collecte près de chez vous via OSM</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_btn, _, _ = st.columns([1, 1, 1])
    with col_btn:
        if st.button("🚀 Commencer", type="primary", use_container_width=True):
            st.session_state.landing_done = True
            st.rerun()

    st.markdown("<hr style='border-color:#1b4332; margin:20px 0 8px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; color:#52b788; font-size:12px'>
        Projet scolaire • Modèle entraîné sur TrashNet & TACO •
        Consignes ADEME / Citeo
    </p>
    """, unsafe_allow_html=True)
    st.stop()

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
        st.image(image, caption="Image analysée", use_column_width=True)

        with st.spinner("🌿 Analyse en cours..."):
            image_data.seek(0)
            result, err = predict_image(image_data)

        if err:
            st.error(f"❌ {err}")
        elif result:
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
            if st.session_state.user and st.session_state.access_token:
                db_save_scan(
                    st.session_state.user["id"],
                    st.session_state.access_token,
                    result["label"], result["bac"], result["confidence"],
                )
                st.success("✅ Scan sauvegardé dans votre historique.")
            else:
                st.markdown(
                    "<div style='color:#95d5b2; font-size:13px; margin-top:4px'>"
                    "💡 <a href='#' style='color:#52b788'>Connectez-vous</a> pour sauvegarder cet historique définitivement.</div>",
                    unsafe_allow_html=True,
                )

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

    type_dechet = st.selectbox("Type de déchet", list(OU_JETER_TYPES.keys()), key="ou_jeter_type")
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
        raw_addr = st.text_input("", placeholder="Ex: 10 rue de Rivoli, Paris",
                                  key="ou_jeter_adresse", label_visibility="collapsed")
        selected_addr = None

    rechercher = st.button("🔍 Rechercher les points de collecte", type="primary", key="ou_jeter_btn")

    if rechercher:
        lat = lon = label_addr = None

        if SEARCHBOX_AVAILABLE:
            if isinstance(selected_addr, dict):
                lat, lon, label_addr = selected_addr["lat"], selected_addr["lon"], selected_addr["label"]
            else:
                st.warning("Sélectionnez une adresse dans la liste déroulante.")
        else:
            if raw_addr.strip():
                with st.spinner("📍 Géolocalisation en cours..."):
                    lat, lon, label_addr = geocode_address(raw_addr)
                if lat is None:
                    st.error("❌ Adresse introuvable.")
            else:
                st.warning("Veuillez entrer une adresse.")

        if lat is not None:
            st.success(f"📍 **{label_addr}**")

            with st.spinner("🗺️ Recherche des points de collecte..."):
                elements, err = search_overpass(lat, lon, config)

            if err:
                st.error(f"❌ {err}")
            else:
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
                    el["_lat"], el["_lon"] = el_lat, el_lon
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
                    st.success(f"✅ {len(results)} point(s) trouvé(s) dans un rayon de {config['radius'] // 1000} km")

                    if FOLIUM_AVAILABLE:
                        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
                        folium.Marker([lat, lon], popup="📍 Votre position",
                                      icon=folium.Icon(color="red", icon="home")).add_to(m)
                        for el in top:
                            tags = el.get("tags", {})
                            name = html_lib.escape(tags.get("name") or tags.get("operator") or "Point de collecte")
                            addr_str = " ".join(html_lib.escape(p) for p in [
                                tags.get("addr:housenumber", ""), tags.get("addr:street", ""),
                                tags.get("addr:city", ""),
                            ] if p)
                            popup_html = (f"<b>{name}</b>"
                                          + (f"<br>{addr_str}" if addr_str else "")
                                          + f"<br><b>📏 {el['_distance']:.1f} km</b>")
                            folium.Marker([el["_lat"], el["_lon"]],
                                          popup=folium.Popup(popup_html, max_width=220),
                                          icon=folium.Icon(color="green", icon="leaf")).add_to(m)
                        components.html(m._repr_html_(), height=320)

                    st.subheader(f"Les {min(10, len(top))} points les plus proches")
                    for i, el in enumerate(top[:10], 1):
                        tags = el.get("tags", {})
                        name = html_lib.escape(tags.get("name") or tags.get("operator") or "Point de collecte")
                        addr_num = html_lib.escape(tags.get("addr:housenumber", ""))
                        addr_street = html_lib.escape(tags.get("addr:street", ""))
                        addr_city = html_lib.escape(tags.get("addr:city", ""))
                        addr_full = " ".join(p for p in [addr_num, addr_street] if p)
                        if addr_city:
                            addr_full = f"{addr_full}, {addr_city}" if addr_full else addr_city
                        opening = html_lib.escape(tags.get("opening_hours", ""))[:100]

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
            <b style='color:#95d5b2'>① Contactez votre mairie</b> (par téléphone ou sur leur site)<br>
            <b style='color:#95d5b2'>② Déclarez votre encombrant</b> (type, taille approximative)<br>
            <b style='color:#95d5b2'>③ Recevez un numéro de passage</b> à scotcher sur l'objet<br>
            <b style='color:#95d5b2'>④ Déposez l'objet sur le trottoir</b> la veille du jour indiqué<br>
            <b style='color:#95d5b2'>⑤ Les agents passent</b> et récupèrent l'objet
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

    user = st.session_state.user
    token = st.session_state.access_token

    if user and token:
        st.caption(f"Historique complet de votre compte — toutes les sessions.")
        with st.spinner("Chargement..."):
            scans_db = db_get_scans(user["id"], token)

        if not scans_db:
            st.markdown("""
            <div style='text-align:center; padding:40px; border:2px dashed #2d6a4f;
                        border-radius:16px; margin-top:20px'>
                <div style='font-size:40px'>🌿</div>
                <p style='color:#95d5b2'>Aucun scan encore. Analysez un déchet !</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            total = len(scans_db)
            conf_moy = sum(s["confidence"] for s in scans_db) / total
            labels_list = [s["label"] for s in scans_db]
            plus_frequent = max(set(labels_list), key=labels_list.count)

            col1, col2, col3 = st.columns(3)
            col1.metric("Scans totaux", total)
            col2.metric("Confiance moyenne", f"{conf_moy:.1f}%")
            col3.metric("Déchet le + scanné", plus_frequent)
            st.divider()

            for scan in scans_db:
                color = BAC_COLORS.get(scan["bac"], "#2d6a4f")
                conf = scan["confidence"]
                conf_color = "#52b788" if conf >= 80 else "#f4a261" if conf >= 60 else "#e63946"
                heure = scan.get("scanned_at", "")[:16].replace("T", " ")
                st.markdown(
                    f"<div style='background:#1b4332; border:1px solid #2d6a4f; border-radius:10px;"
                    f"padding:14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center'>"
                    f"<span style='color:#95d5b2; font-size:13px'>🕐 {heure}</span>"
                    f"<span style='color:#d8f3dc; font-weight:bold'>{scan['label']}</span>"
                    f"<span style='background:{color}; color:white; padding:4px 12px; border-radius:20px; font-size:13px'>{scan['bac']}</span>"
                    f"<span style='color:{conf_color}; font-weight:bold'>{conf}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            if st.button("🗑️ Effacer mon historique", type="secondary"):
                db_clear_scans(user["id"], token)
                st.rerun()
    else:
        st.caption("Session en cours uniquement — connectez-vous pour sauvegarder définitivement.")
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
            labels_list = [s["label"] for s in st.session_state.historique]
            plus_frequent = max(set(labels_list), key=labels_list.count)

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
                    f"padding:14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center'>"
                    f"<span style='color:#95d5b2; font-size:13px'>🕐 {scan['heure']}</span>"
                    f"<span style='color:#d8f3dc; font-weight:bold'>{scan['label']}</span>"
                    f"<span style='background:{color}; color:white; padding:4px 12px; border-radius:20px; font-size:13px'>{scan['bac']}</span>"
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

    for cat in CATEGORIES_DISPLAY:
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
    st.subheader("Hors périmètre")
    st.warning("Utilisez l'onglet **Où jeter ?** pour ces catégories.")
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
    st.subheader("Limites connues")
    st.markdown("""
    | Situation | Impact | Recommandation |
    |-----------|--------|----------------|
    | Objet sur fond complexe | Précision réduite | Photographier sur fond neutre |
    | Plusieurs déchets | Classification incorrecte | Un seul objet par photo |
    | Mauvaise luminosité | Précision réduite | Bonne lumière naturelle |
    | Score < 60% | Résultat peu fiable | Vérifier manuellement |
    """)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset principal", "TrashNet")
        st.caption("~2 500 images, 6 classes")
        st.metric("Dataset complémentaire", "TACO")
        st.caption("Photos en contexte réel")
    with col2:
        st.metric("Architecture", "EfficientNet-B2")
        st.caption("Transfer learning — ImageNet → déchets")
        st.metric("Référentiel", "ADEME / Citeo")
        st.caption("Consignes de tri officielles France")
