import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import math
import requests
import os
import html as html_lib
from datetime import datetime
from pathlib import Path

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from torchvision import models

try:
    from supabase import create_client as _sb_create
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "waste_ai_v4.pt"
NUM_CLASSES = 6
IMG_SIZE = 260
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000

_SB_URL = os.environ.get("SUPABASE_URL", "")
_SB_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
DB_READY = SUPABASE_AVAILABLE and bool(_SB_URL) and bool(_SB_KEY)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CATEGORIES = {
    0: {"label": "Carton",    "bac": "Bac jaune",                    "emoji": "📦", "consigne": "Aplatissez le carton et déposez-le dans le bac jaune."},
    1: {"label": "Verre",     "bac": "Colonne à verre",              "emoji": "🍾", "consigne": "Déposez dans une colonne à verre. Retirez les couvercles."},
    2: {"label": "Métal",     "bac": "Bac jaune",                    "emoji": "🥫", "consigne": "Déposez dans le bac jaune. Écrasez les canettes si possible."},
    3: {"label": "Papier",    "bac": "Bac jaune",                    "emoji": "📄", "consigne": "Déposez dans le bac jaune. Pas de papier gras ni de mouchoirs."},
    4: {"label": "Plastique", "bac": "Bac jaune",                    "emoji": "🧴", "consigne": "Déposez dans le bac jaune. Videz et rincez les emballages."},
    5: {"label": "Résidus",   "bac": "Bac gris (ordures ménagères)", "emoji": "🗑️", "consigne": "Déposez dans le bac gris. Cet objet n'est pas recyclable."},
}

BAC_COLORS = {
    "Bac jaune":                    "#f5c518",
    "Colonne à verre":              "#2d6a4f",
    "Bac gris (ordures ménagères)": "#616161",
}

OU_JETER_TYPES = {
    "🔋 Piles & Batteries":         {"queries": ['node["amenity"="recycling"]["recycling:batteries"="yes"](around:{r},{lat},{lon});'], "radius": 5000,  "conseil": "Bacs de collecte en supermarché, bureau de tabac.", "eco_org": "Corepile / Screlec"},
    "📱 Électronique & DEEE":       {"queries": ['node["amenity"="recycling"]["recycling:electronics"="yes"](around:{r},{lat},{lon});', 'node["amenity"="recycling"]["recycling_type"="centre"](around:{r},{lat},{lon});'], "radius": 10000, "conseil": "Le vendeur est obligé de reprendre votre ancien appareil (loi AGEC).", "eco_org": "Ecosystem / Ecologic"},
    "👕 Vêtements & Textiles":      {"queries": ['node["amenity"="recycling"]["recycling:clothes"="yes"](around:{r},{lat},{lon});'], "radius": 5000,  "conseil": "Bornes de collecte en ville, parking de supermarché.", "eco_org": "Le Relais / Emmaüs"},
    "💊 Médicaments":               {"queries": ['node["amenity"="pharmacy"](around:{r},{lat},{lon});'], "radius": 3000, "conseil": "Rapportez vos médicaments non utilisés en pharmacie.", "eco_org": "Cyclamed"},
    "🛋️ Encombrants & Déchetterie": {"queries": ['node["amenity"="recycling"]["recycling_type"="centre"](around:{r},{lat},{lon});', 'node["amenity"="waste_transfer_station"](around:{r},{lat},{lon});'], "radius": 15000, "conseil": "Déchetterie pour meubles, matelas, électroménager volumineux.", "eco_org": "Mairie / Collectivité"},
}

_OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# ── Modèle ─────────────────────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
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
        _model = m
        return m
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
        return None

load_model()

# ── Supabase ───────────────────────────────────────────────────────────────────
def _authed_client(token):
    client = _sb_create(_SB_URL, _SB_KEY)
    client.postgrest.auth(token)
    return client

def auth_login(email, password):
    try:
        client = _sb_create(_SB_URL, _SB_KEY)
        r = client.auth.sign_in_with_password({"email": email, "password": password})
        return r.user, r.session, None
    except Exception:
        return None, None, "Email ou mot de passe incorrect."

def auth_signup(email, password):
    try:
        client = _sb_create(_SB_URL, _SB_KEY)
        r = client.auth.sign_up({"email": email, "password": password})
        return r.user, r.session, None
    except Exception as e:
        return None, None, str(e)

def db_save_scan(user_id, token, label, bac, confidence):
    try:
        _authed_client(token).table("scans").insert({
            "user_id": user_id, "label": label, "bac": bac, "confidence": confidence,
        }).execute()
    except Exception:
        pass

def db_get_scans(user_id, token):
    try:
        r = (_authed_client(token).table("scans")
             .select("label, bac, confidence, scanned_at")
             .eq("user_id", user_id)
             .order("scanned_at", desc=True)
             .limit(100).execute())
        return r.data or []
    except Exception:
        return []

def db_clear_scans(user_id, token):
    try:
        _authed_client(token).table("scans").delete().eq("user_id", user_id).execute()
    except Exception:
        pass

# ── Géolocalisation ────────────────────────────────────────────────────────────
def geocode_address(address):
    try:
        r = requests.get("https://api-adresse.data.gouv.fr/search/", params={"q": address, "limit": 1}, timeout=10)
        features = r.json().get("features", [])
        if not features:
            return None, None, None
        coords = features[0]["geometry"]["coordinates"]
        return coords[1], coords[0], features[0]["properties"]["label"]
    except Exception:
        return None, None, None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def search_overpass(lat, lon, config):
    parts = [q.format(r=config["radius"], lat=lat, lon=lon) for q in config["queries"]]
    query = f"[out:json][timeout:30];({''.join(parts)});out center;"
    for ep in _OVERPASS_ENDPOINTS:
        try:
            r = requests.get(ep, params={"data": query}, timeout=35)
            if r.status_code == 200:
                return r.json().get("elements", []), None
        except Exception:
            continue
    return [], "Impossible de contacter OpenStreetMap."

# ── Fonctions principales ──────────────────────────────────────────────────────
def predict(image, user_state):
    if image is None:
        return "<p style='color:#95d5b2;text-align:center;padding:30px;font-size:16px'>📸 Prenez une photo ou uploadez une image pour commencer.</p>"

    model = load_model()
    if model is None:
        return "<p style='color:#e63946'>❌ Modèle non disponible (fichier waste_ai_v4.pt manquant).</p>"

    try:
        img = Image.fromarray(image).convert("RGB")
    except Exception:
        return "<p style='color:#e63946'>❌ Image invalide.</p>"

    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        proba = torch.softmax(output, dim=1)
        confidence, class_id = torch.max(proba, dim=1)

    cat = CATEGORIES[class_id.item()]
    conf = round(confidence.item() * 100, 1)
    color = BAC_COLORS.get(cat["bac"], "#2d6a4f")
    conf_color = "#52b788" if conf >= 80 else "#f4a261" if conf >= 60 else "#e63946"

    if DB_READY and user_state.get("id") and user_state.get("token"):
        db_save_scan(user_state["id"], user_state["token"], cat["label"], cat["bac"], conf)

    warning = ""
    if conf < 60:
        warning = "<div style='background:#3d2b00;border-left:4px solid #f4a261;border-radius:8px;padding:12px;margin-top:12px;color:#f4a261'>⚠️ Confiance faible — essayez sur fond neutre avec un seul objet.</div>"

    return f"""
<div>
    <div style='display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap'>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:12px;padding:16px;flex:1;min-width:130px;text-align:center'>
            <div style='color:#95d5b2;font-size:12px;margin-bottom:6px'>DÉCHET DÉTECTÉ</div>
            <div style='color:#d8f3dc;font-size:20px;font-weight:bold'>{cat['emoji']} {cat['label']}</div>
        </div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:12px;padding:16px;flex:1;min-width:130px;text-align:center'>
            <div style='color:#95d5b2;font-size:12px;margin-bottom:6px'>CONFIANCE</div>
            <div style='color:{conf_color};font-size:22px;font-weight:bold'>{conf}%</div>
        </div>
    </div>
    <div style='background:{color};color:{"#000" if color=="#f5c518" else "white"};padding:16px;border-radius:12px;font-size:18px;font-weight:bold;text-align:center;margin-bottom:12px'>
        🗑️ {cat['bac']}
    </div>
    <div style='background:#1b4332;border-left:4px solid #52b788;border-radius:8px;padding:12px;color:#d8f3dc'>
        💡 {cat['consigne']}
    </div>
    {warning}
</div>"""

def rechercher_points(type_dechet, adresse):
    if not adresse.strip():
        return "<p style='color:#f4a261'>⚠️ Entrez une adresse.</p>"
    lat, lon, label_addr = geocode_address(adresse)
    if lat is None:
        return "<p style='color:#e63946'>❌ Adresse introuvable. Essayez avec le nom de ville.</p>"
    config = OU_JETER_TYPES[type_dechet]
    elements, err = search_overpass(lat, lon, config)
    if err:
        return f"<p style='color:#e63946'>❌ {err}</p>"

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
        el["_dist"] = haversine_km(lat, lon, el_lat, el_lon)
        results.append(el)

    results.sort(key=lambda x: x["_dist"])
    top = results[:10]

    if not top:
        return f"<p style='color:#f4a261'>😕 Aucun point trouvé dans un rayon de {config['radius']//1000} km.</p>"

    html = f"""
<div style='background:#1b4332;border-left:4px solid #52b788;border-radius:8px;padding:12px;margin-bottom:12px'>
    <b style='color:#d8f3dc'>📍 {html_lib.escape(label_addr)}</b> —
    <span style='color:#52b788'>{len(results)} point(s) trouvé(s) dans {config['radius']//1000} km</span><br>
    <span style='color:#95d5b2;font-size:13px'>💡 {config['conseil']} | {config['eco_org']}</span>
</div>"""

    for i, el in enumerate(top, 1):
        tags = el.get("tags", {})
        name = html_lib.escape(tags.get("name") or tags.get("operator") or "Point de collecte")
        addr_parts = [tags.get("addr:housenumber",""), tags.get("addr:street",""), tags.get("addr:city","")]
        addr_str = html_lib.escape(" ".join(p for p in addr_parts if p))
        html += f"""
<div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:14px;margin:8px 0;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <div>
        <b style='color:#d8f3dc'>{i}. {name}</b>
        {f"<br><span style='color:#95d5b2;font-size:13px'>📍 {addr_str}</span>" if addr_str else ""}
    </div>
    <span style='background:#2d6a4f;color:#d8f3dc;padding:4px 14px;border-radius:20px;font-size:13px;white-space:nowrap'>📏 {el['_dist']:.1f} km</span>
</div>"""
    return html

def do_login(email, password):
    if not DB_READY:
        return {}, gr.update(value="❌ Secrets Supabase manquants dans les paramètres du Space."), gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})
    user, session, err = auth_login(email, password)
    if err:
        return {}, gr.update(value=f"❌ {err}"), gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})
    state = {"id": str(user.id), "email": user.email, "token": session.access_token}
    return state, gr.update(value=""), gr.update(visible=False), gr.update(visible=True), gr.update(value=navbar_html(user.email)), get_historique(state)

def do_signup(email, password, confirm):
    if not DB_READY:
        return {}, gr.update(value="❌ Secrets Supabase manquants dans les paramètres du Space."), gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})
    if password != confirm:
        return {}, gr.update(value="❌ Les mots de passe ne correspondent pas."), gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})
    if len(password) < 6:
        return {}, gr.update(value="❌ Mot de passe trop court (6 caractères min)."), gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})
    user, session, err = auth_signup(email, password)
    if err:
        return {}, gr.update(value=f"❌ {err}"), gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})
    if session:
        state = {"id": str(user.id), "email": user.email, "token": session.access_token}
        return state, gr.update(value=""), gr.update(visible=False), gr.update(visible=True), gr.update(value=navbar_html(user.email)), get_historique(state)
    return {}, gr.update(value="✅ Compte créé ! Connectez-vous."), gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})

def do_logout(user_state):
    return {}, gr.update(visible=True), gr.update(visible=False), gr.update(value=navbar_html(None)), get_historique({})

def get_historique(user_state):
    if not user_state or not user_state.get("id"):
        return "<div style='text-align:center;padding:40px;border:2px dashed #2d6a4f;border-radius:16px;margin:20px 0'><div style='font-size:40px'>🔒</div><p style='color:#95d5b2;margin:8px 0'>Connectez-vous pour voir votre historique persistant.</p></div>"
    scans = db_get_scans(user_state["id"], user_state["token"])
    if not scans:
        return "<div style='text-align:center;padding:40px;border:2px dashed #2d6a4f;border-radius:16px;margin:20px 0'><div style='font-size:40px'>🌿</div><p style='color:#95d5b2;margin:8px 0'>Aucun scan encore. Analysez un déchet !</p></div>"

    conf_moy = sum(s["confidence"] for s in scans) / len(scans)
    labels_list = [s["label"] for s in scans]
    plus_frequent = max(set(labels_list), key=labels_list.count)

    html = f"""
<div style='display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap'>
    <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:12px;padding:14px;flex:1;min-width:100px;text-align:center'>
        <div style='color:#95d5b2;font-size:12px'>SCANS TOTAUX</div>
        <div style='color:#d8f3dc;font-size:22px;font-weight:bold'>{len(scans)}</div>
    </div>
    <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:12px;padding:14px;flex:1;min-width:100px;text-align:center'>
        <div style='color:#95d5b2;font-size:12px'>CONFIANCE MOY.</div>
        <div style='color:#52b788;font-size:22px;font-weight:bold'>{conf_moy:.1f}%</div>
    </div>
    <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:12px;padding:14px;flex:1;min-width:100px;text-align:center'>
        <div style='color:#95d5b2;font-size:12px'>LE + SCANNÉ</div>
        <div style='color:#d8f3dc;font-size:18px;font-weight:bold'>{plus_frequent}</div>
    </div>
</div>"""

    for s in scans:
        color = BAC_COLORS.get(s["bac"], "#2d6a4f")
        conf = s["confidence"]
        conf_color = "#52b788" if conf >= 80 else "#f4a261" if conf >= 60 else "#e63946"
        heure = s.get("scanned_at", "")[:16].replace("T", " ")
        html += f"""
<div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <span style='color:#95d5b2;font-size:13px'>🕐 {heure}</span>
    <span style='color:#d8f3dc;font-weight:bold'>{s['label']}</span>
    <span style='background:{color};color:{"#000" if color=="#f5c518" else "white"};padding:3px 10px;border-radius:20px;font-size:13px'>{s['bac']}</span>
    <span style='color:{conf_color};font-weight:bold'>{conf}%</span>
</div>"""
    return html

def clear_historique(user_state):
    if user_state and user_state.get("id"):
        db_clear_scans(user_state["id"], user_state["token"])
    return get_historique(user_state)

def navbar_html(email):
    if email:
        short = email.split("@")[0]
        return f"<div style='text-align:right;padding:8px 0'><span style='color:#95d5b2;font-size:13px'>👤 {short}</span></div>"
    return ""

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ── Fond global ── */
body, .gradio-container, .main, footer, .app {
    background: linear-gradient(160deg, #0d1f14 0%, #1a2e1e 100%) !important;
}
footer { display: none !important; }

/* ── Largeur PC uniquement ── */
@media (min-width: 768px) {
    .gradio-container { max-width: 1100px !important; width: 90% !important; margin: 0 auto !important; }
    .contain { max-width: 1100px !important; }
}

/* ── Onglets ── */
[role="tablist"] {
    background: transparent !important;
    border-bottom: 2px solid #2d6a4f !important;
    gap: 2px !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
}
[role="tab"] {
    background: transparent !important; color: #95d5b2 !important;
    font-weight: 600 !important; font-size: 14px !important;
    border: none !important; border-radius: 8px 8px 0 0 !important;
    padding: 10px 16px !important; white-space: nowrap !important;
}
[role="tab"]:hover { background: #1b4332 !important; color: #d8f3dc !important; }
[role="tab"][aria-selected="true"] {
    background: #1b4332 !important; color: #d8f3dc !important;
    border-bottom: 3px solid #52b788 !important;
}

/* ── Tous les blocs / conteneurs → fond sombre ── */
.block, .panel, .tabitem, [data-testid="block"],
.form, .gap, .gr-group, .group,
div.svelte-vt1mxs, div.svelte-1f354aw,
section, article {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Groupes → fond vert sombre ── */
.gr-group > div, [data-testid="group"] > div,
.gradio-group, fieldset {
    background: #132d1e !important;
    border: 1px solid #2d6a4f !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* ── Textes ── */
h1, h2, h3, h4 { color: #d8f3dc !important; }
label, p, span, li, .label-wrap span, .prose { color: #b7e4c7 !important; }

/* ── Boutons ── */
button.primary, button[class*="primary"] {
    background: linear-gradient(135deg, #2d6a4f, #52b788) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: bold !important;
}
button.secondary, button[class*="secondary"] {
    background: transparent !important;
    border: 1px solid #52b788 !important;
    color: #95d5b2 !important; border-radius: 10px !important;
}

/* ── Inputs ── */
input, input[type=text], input[type=email], input[type=password], textarea {
    background: #1b4332 !important; color: #d8f3dc !important;
    border: 1px solid #2d6a4f !important; border-radius: 10px !important;
}
input::placeholder, textarea::placeholder { color: #52b788 !important; }

/* ── Select / Dropdown ── */
select, .choices, .dropdown-arrow {
    background: #1b4332 !important; color: #d8f3dc !important;
    border: 1px solid #2d6a4f !important; border-radius: 10px !important;
}
ul[role="listbox"], [role="option"] { background: #1b4332 !important; color: #d8f3dc !important; }
[role="option"]:hover { background: #2d6a4f !important; }

/* ── Zone image/upload ── */
.image-frame, .upload-container, [data-testid="image"],
.image-container, .svelte-p3y7hu {
    background: #1b4332 !important;
    border: 2px dashed #52b788 !important;
    border-radius: 12px !important;
}

/* ── Séparateur ── */
hr { border-color: #2d6a4f !important; }

/* ── Markdown ── */
.prose * { color: #b7e4c7 !important; }
</style>
"""

# ── Interface ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Waste AI ♻️") as demo:

    user_state = gr.State({})

    # CSS global
    gr.HTML(CSS)

    # ── Navbar ──────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            gr.HTML("""
            <div style='display:flex;align-items:center;gap:14px;padding:16px 0 8px 0'>
                <span style='font-size:48px;line-height:1'>♻️</span>
                <div>
                    <div style='color:#d8f3dc;font-size:28px;font-weight:bold;font-family:Georgia,serif;line-height:1.2'>Waste AI</div>
                    <div style='color:#95d5b2;font-size:13px'>Photographiez un déchet — l'IA vous dit où le jeter.</div>
                </div>
            </div>""")
        with gr.Column(scale=1, min_width=160):
            navbar_status = gr.HTML("")
            logout_btn = gr.Button("🚪 Déconnexion", variant="secondary", size="sm", visible=False)

    gr.HTML("<hr style='border-color:#2d6a4f;margin:0 0 8px 0'>")

    # ── Tabs ────────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1 : Analyser ─────────────────────────────────────────────────
        with gr.Tab("🔍 Analyser"):
            image_input = gr.Image(sources=["webcam", "upload"], type="numpy", label="Photo du déchet")
            analyse_btn = gr.Button("🌿 Analyser", variant="primary", size="lg")
            result_html = gr.HTML("<div style='text-align:center;padding:30px;border:2px dashed #2d6a4f;border-radius:16px;margin-top:12px'><div style='font-size:48px'>📸</div><p style='color:#95d5b2;margin:8px 0'>Prenez une photo ou uploadez une image pour commencer.</p></div>")
            analyse_btn.click(fn=predict, inputs=[image_input, user_state], outputs=result_html)

        # ── Tab 2 : Où jeter ? ───────────────────────────────────────────────
        with gr.Tab("🗺️ Où jeter ?"):
            gr.HTML("<p style='color:#95d5b2;margin:0 0 12px 0'>Pour les déchets complexes — sélectionnez le type et entrez votre adresse.</p>")
            type_dechet = gr.Dropdown(choices=list(OU_JETER_TYPES.keys()), label="Type de déchet", value="🔋 Piles & Batteries")
            adresse_input = gr.Textbox(placeholder="Ex: 10 rue de Rivoli, Paris", label="Votre adresse")
            search_btn = gr.Button("🔍 Rechercher les points de collecte", variant="primary")
            map_result = gr.HTML()
            search_btn.click(fn=rechercher_points, inputs=[type_dechet, adresse_input], outputs=map_result)

        # ── Tab 3 : Historique ───────────────────────────────────────────────
        with gr.Tab("📋 Historique") as historique_tab:
            with gr.Row():
                refresh_btn = gr.Button("🔄 Rafraîchir", variant="secondary", size="sm")
                clear_btn   = gr.Button("🗑️ Effacer", variant="secondary", size="sm")
            historique_html = gr.HTML()
            refresh_btn.click(fn=get_historique, inputs=[user_state], outputs=historique_html)
            clear_btn.click(fn=clear_historique, inputs=[user_state], outputs=historique_html)
            # Rafraîchit quand on sélectionne l'onglet
            historique_tab.select(fn=get_historique, inputs=[user_state], outputs=historique_html)

        # ── Tab 4 : Compte ───────────────────────────────────────────────────
        with gr.Tab("👤 Compte"):
            auth_msg = gr.Markdown("")

            with gr.Group(visible=True) as login_group:
                gr.HTML("<h3 style='color:#95d5b2;margin:0 0 12px 0'>Se connecter</h3>")
                login_email = gr.Textbox(label="Email", type="email")
                login_pw    = gr.Textbox(label="Mot de passe", type="password")
                login_btn   = gr.Button("Se connecter", variant="primary")
                gr.HTML("<hr style='border-color:#2d6a4f;margin:16px 0'>")
                gr.HTML("<h3 style='color:#95d5b2;margin:0 0 12px 0'>Créer un compte</h3>")
                signup_email = gr.Textbox(label="Email", type="email")
                signup_pw    = gr.Textbox(label="Mot de passe", type="password")
                signup_pw2   = gr.Textbox(label="Confirmer le mot de passe", type="password")
                signup_btn   = gr.Button("Créer mon compte", variant="primary")

            with gr.Group(visible=False) as logged_group:
                gr.HTML("<p style='color:#52b788'>✅ Vous êtes connecté. Vos scans sont sauvegardés automatiquement.</p>")

            login_btn.click(
                fn=do_login,
                inputs=[login_email, login_pw],
                outputs=[user_state, auth_msg, login_group, logged_group, navbar_status, historique_html]
            )
            signup_btn.click(
                fn=do_signup,
                inputs=[signup_email, signup_pw, signup_pw2],
                outputs=[user_state, auth_msg, login_group, logged_group, navbar_status, historique_html]
            )
            logout_btn.click(
                fn=do_logout,
                inputs=[user_state],
                outputs=[user_state, login_group, logged_group, navbar_status, historique_html]
            )

            # Afficher/cacher le bouton déconnexion selon état
            login_btn.click(fn=lambda s: gr.update(visible=bool(s.get("id"))), inputs=[user_state], outputs=[logout_btn])
            signup_btn.click(fn=lambda s: gr.update(visible=bool(s.get("id"))), inputs=[user_state], outputs=[logout_btn])
            logout_btn.click(fn=lambda: gr.update(visible=False), outputs=[logout_btn])

        # ── Tab 5 : Couverture ───────────────────────────────────────────────
        with gr.Tab("ℹ️ Couverture"):
            gr.HTML("""
<div>
    <h3 style='color:#95d5b2'>Catégories couvertes (6)</h3>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px'>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px'><b style='color:#f5c518'>📦 Carton</b><p style='color:#95d5b2;font-size:13px;margin:4px 0'>Bac jaune — Aplatissez avant de déposer.</p></div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px'><b style='color:#2d9e6b'>🍾 Verre</b><p style='color:#95d5b2;font-size:13px;margin:4px 0'>Colonne à verre — Retirez les couvercles.</p></div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px'><b style='color:#f5c518'>🥫 Métal</b><p style='color:#95d5b2;font-size:13px;margin:4px 0'>Bac jaune — Écrasez les canettes.</p></div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px'><b style='color:#f5c518'>📄 Papier</b><p style='color:#95d5b2;font-size:13px;margin:4px 0'>Bac jaune — Pas de papier gras.</p></div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px'><b style='color:#f5c518'>🧴 Plastique</b><p style='color:#95d5b2;font-size:13px;margin:4px 0'>Bac jaune — Videz et rincez.</p></div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px'><b style='color:#aaa'>🗑️ Résidus</b><p style='color:#95d5b2;font-size:13px;margin:4px 0'>Bac gris — Non recyclable.</p></div>
    </div>
    <hr style='border-color:#2d6a4f'>
    <h3 style='color:#95d5b2'>Limites connues</h3>
    <ul style='color:#b7e4c7'>
        <li>Un seul objet par photo</li>
        <li>Fond neutre recommandé</li>
        <li>Bonne luminosité nécessaire</li>
        <li>Score &lt; 60% = résultat peu fiable</li>
    </ul>
    <hr style='border-color:#2d6a4f'>
    <div style='display:flex;gap:12px;flex-wrap:wrap'>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px;flex:1;min-width:120px'>
            <div style='color:#95d5b2;font-size:12px'>ARCHITECTURE</div>
            <div style='color:#d8f3dc;font-weight:bold'>EfficientNet-B2</div>
            <div style='color:#95d5b2;font-size:12px'>Transfer learning ImageNet</div>
        </div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px;flex:1;min-width:120px'>
            <div style='color:#95d5b2;font-size:12px'>DATASETS</div>
            <div style='color:#d8f3dc;font-weight:bold'>TrashNet + TACO</div>
            <div style='color:#95d5b2;font-size:12px'>~2500 images annotées</div>
        </div>
        <div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;padding:12px;flex:1;min-width:120px'>
            <div style='color:#95d5b2;font-size:12px'>RÉFÉRENTIEL</div>
            <div style='color:#d8f3dc;font-weight:bold'>ADEME / Citeo</div>
            <div style='color:#95d5b2;font-size:12px'>Consignes de tri France</div>
        </div>
    </div>
    <p style='text-align:center;color:#52b788;font-size:12px;margin-top:20px'>Projet scolaire • Semestre 2026 • EfficientNet-B2 • TrashNet & TACO</p>
</div>""")

demo.launch()
