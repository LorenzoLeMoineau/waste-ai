import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io, math, requests, os, html as html_lib
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

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Waste AI 📱", page_icon="♻️",
                   layout="centered", initial_sidebar_state="collapsed")

Image.MAX_IMAGE_PIXELS = 50_000_000
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MODEL_PATH = Path(__file__).parent / "waste_ai_v4.pt"
NUM_CLASSES = 6
IMG_SIZE = 260

_SB_URL = os.environ.get("SUPABASE_URL", "")
_SB_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
DB_READY = SUPABASE_AVAILABLE and bool(_SB_URL) and bool(_SB_KEY)

# ── CSS Mobile ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(160deg,#0d1f14 0%,#1a2e1e 100%) !important; }
[data-testid="stHeader"], [data-testid="stDecoration"] { display:none !important; }
.block-container { padding: 0.8rem 0.8rem 2rem 0.8rem !important; max-width: 480px !important; margin: 0 auto !important; }
h1 { color:#d8f3dc !important; font-family:Georgia,serif !important; font-size:1.8rem !important; }
h2,h3 { color:#95d5b2 !important; }
p, label, .stMarkdown { color:#b7e4c7 !important; }
[data-testid="stTabs"] [role="tab"] { color:#95d5b2 !important; font-weight:600 !important; font-size:13px !important; padding:6px 10px !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background:#1b4332 !important; color:#d8f3dc !important; border-bottom:3px solid #52b788 !important; }
[data-testid="baseButton-primary"] { background:linear-gradient(135deg,#2d6a4f,#52b788) !important; border:none !important; color:white !important; font-weight:bold !important; border-radius:12px !important; width:100% !important; padding:0.6rem !important; }
[data-testid="baseButton-secondary"] { background:transparent !important; border:1px solid #52b788 !important; color:#95d5b2 !important; border-radius:12px !important; }
[data-testid="stFileUploader"], [data-testid="stCameraInput"] { background:#1b4332 !important; border:2px dashed #52b788 !important; border-radius:14px !important; }
[data-testid="stExpander"] { background:#1b4332 !important; border:1px solid #2d6a4f !important; border-radius:12px !important; }
[data-testid="metric-container"] { background:#1b4332 !important; border:1px solid #2d6a4f !important; border-radius:12px !important; padding:12px !important; }
[data-testid="stMetricLabel"] { color:#95d5b2 !important; font-size:12px !important; }
[data-testid="stMetricValue"] { color:#d8f3dc !important; font-size:1.2rem !important; }
[data-testid="stInfo"] { background:#1b4332 !important; border-left:4px solid #52b788 !important; color:#d8f3dc !important; }
[data-testid="stSuccess"] { background:#1b4332 !important; border-left:4px solid #52b788 !important; }
[data-testid="stWarning"] { background:#3d2b00 !important; border-left:4px solid #f4a261 !important; }
[data-testid="stError"] { background:#3d0000 !important; border-left:4px solid #e63946 !important; }
input, textarea, select { background:#1b4332 !important; color:#d8f3dc !important; border:1px solid #2d6a4f !important; border-radius:10px !important; }
hr { border-color:#2d6a4f !important; }
div[class*="searchbox"] input { background:#1b4332 !important; color:#d8f3dc !important; border:1px solid #2d6a4f !important; border-radius:10px !important; }
iframe { border-radius:12px !important; overflow:hidden; }
[data-testid="stCustomComponentV1"] { height:280px !important; min-height:unset !important; overflow:hidden !important; }
</style>
""", unsafe_allow_html=True)

# ── Modèle ─────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

CATEGORIES = {
    0: {"label":"Carton",    "bac":"Bac jaune",                    "emoji":"📦", "consigne":"Aplatissez le carton et déposez dans le bac jaune."},
    1: {"label":"Verre",     "bac":"Colonne à verre",              "emoji":"🍾", "consigne":"Déposez dans une colonne à verre. Retirez les couvercles."},
    2: {"label":"Métal",     "bac":"Bac jaune",                    "emoji":"🥫", "consigne":"Déposez dans le bac jaune. Écrasez les canettes si possible."},
    3: {"label":"Papier",    "bac":"Bac jaune",                    "emoji":"📄", "consigne":"Déposez dans le bac jaune. Pas de papier gras."},
    4: {"label":"Plastique", "bac":"Bac jaune",                    "emoji":"🧴", "consigne":"Déposez dans le bac jaune. Videz et rincez."},
    5: {"label":"Résidus",   "bac":"Bac gris (ordures ménagères)", "emoji":"🗑️", "consigne":"Déposez dans le bac gris. Non recyclable."},
}

BAC_COLORS = {
    "Bac jaune":"#f5c518",
    "Colonne à verre":"#2d6a4f",
    "Bac gris (ordures ménagères)":"#616161",
}

OU_JETER_TYPES = {
    "🔋 Piles & Batteries":         {"queries":['node["amenity"="recycling"]["recycling:batteries"="yes"](around:{r},{lat},{lon});'],"radius":5000,"conseil":"Bacs en supermarché, bureau de tabac.","eco_org":"Corepile / Screlec"},
    "📱 Électronique & DEEE":       {"queries":['node["amenity"="recycling"]["recycling:electronics"="yes"](around:{r},{lat},{lon});','node["amenity"="recycling"]["recycling_type"="centre"](around:{r},{lat},{lon});'],"radius":10000,"conseil":"Le vendeur doit reprendre votre ancien appareil (loi AGEC).","eco_org":"Ecosystem / Ecologic"},
    "👕 Vêtements & Textiles":      {"queries":['node["amenity"="recycling"]["recycling:clothes"="yes"](around:{r},{lat},{lon});'],"radius":5000,"conseil":"Bornes en ville, parking de supermarché.","eco_org":"Le Relais / Emmaüs"},
    "💊 Médicaments":               {"queries":['node["amenity"="pharmacy"](around:{r},{lat},{lon});'],"radius":3000,"conseil":"Rapportez en pharmacie (réseau Cyclamed).","eco_org":"Cyclamed"},
    "🛋️ Encombrants & Déchetterie": {"queries":['node["amenity"="recycling"]["recycling_type"="centre"](around:{r},{lat},{lon});','node["amenity"="waste_transfer_station"](around:{r},{lat},{lon});'],"radius":15000,"conseil":"Déchetterie pour meubles, matelas, électroménager.","eco_org":"Mairie / Collectivité"},
}

_OVERPASS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

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

# ── Supabase ───────────────────────────────────────────────────────────────────
def _client(token=None):
    c = _sb_create(_SB_URL, _SB_KEY)
    if token:
        c.postgrest.auth(token)
    return c

def auth_login(email, password):
    try:
        r = _client().auth.sign_in_with_password({"email":email,"password":password})
        return r.user, r.session, None
    except:
        return None, None, "Email ou mot de passe incorrect."

def auth_signup(email, password):
    try:
        r = _client().auth.sign_up({"email":email,"password":password})
        return r.user, r.session, None
    except Exception as e:
        return None, None, str(e)

def db_save(uid, tok, label, bac, conf):
    try:
        _client(tok).table("scans").insert({"user_id":uid,"label":label,"bac":bac,"confidence":conf}).execute()
    except: pass

def db_get(uid, tok):
    try:
        r = _client(tok).table("scans").select("label,bac,confidence,scanned_at").eq("user_id",uid).order("scanned_at",desc=True).limit(100).execute()
        return r.data or []
    except: return []

def db_clear(uid, tok):
    try: _client(tok).table("scans").delete().eq("user_id",uid).execute()
    except: pass

# ── Géoloc ─────────────────────────────────────────────────────────────────────
def search_addr(q):
    if len(q) < 3: return []
    try:
        r = requests.get("https://api-adresse.data.gouv.fr/search/", params={"q":q,"limit":5}, timeout=5)
        return [(f["properties"]["label"],{"lat":f["geometry"]["coordinates"][1],"lon":f["geometry"]["coordinates"][0],"label":f["properties"]["label"]}) for f in r.json().get("features",[])]
    except: return []

def geocode(addr):
    try:
        r = requests.get("https://api-adresse.data.gouv.fr/search/", params={"q":addr,"limit":1}, timeout=8)
        fs = r.json().get("features",[])
        if not fs: return None,None,None
        c = fs[0]["geometry"]["coordinates"]
        return c[1],c[0],fs[0]["properties"]["label"]
    except: return None,None,None

def haversine(la1,lo1,la2,lo2):
    R=6371; dlat=math.radians(la2-la1); dlon=math.radians(lo2-lo1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def search_overpass(lat,lon,config):
    parts=[q.format(r=config["radius"],lat=lat,lon=lon) for q in config["queries"]]
    query=f"[out:json][timeout:30];({''.join(parts)});out center;"
    for ep in _OVERPASS:
        try:
            r=requests.get(ep,params={"data":query},timeout=35)
            if r.status_code==200: return r.json().get("elements",[]),None
        except: continue
    return [],"Impossible de contacter OpenStreetMap."

# ── Session state ──────────────────────────────────────────────────────────────
for k,v in [("user",None),("token",None),("last_img_id",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:16px 0 4px 0'>
    <span style='font-size:44px'>♻️</span>
    <h1 style='margin:4px 0;font-size:1.8rem;color:#d8f3dc;font-family:Georgia,serif'>Waste AI</h1>
    <p style='color:#95d5b2;font-size:13px;margin:0'>Photographiez un déchet — l'IA vous dit où le jeter.</p>
</div>
""", unsafe_allow_html=True)

# ── Auth compacte ──────────────────────────────────────────────────────────────
if st.session_state.user:
    email_short = st.session_state.user["email"].split("@")[0]
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"<p style='color:#52b788;margin:4px 0;font-size:13px'>👤 {email_short}</p>", unsafe_allow_html=True)
    with col2:
        if st.button("Déco.", type="secondary", use_container_width=True):
            st.session_state.user = None
            st.session_state.token = None
            st.rerun()
elif DB_READY:
    with st.expander("👤 Connexion / Créer un compte"):
        mode = st.radio("", ["Se connecter","Créer un compte"], horizontal=True, label_visibility="collapsed")
        email = st.text_input("Email", key="auth_email")
        pwd   = st.text_input("Mot de passe", type="password", key="auth_pwd")
        if mode == "Créer un compte":
            pwd2 = st.text_input("Confirmer", type="password", key="auth_pwd2")
        if st.button("Valider", type="primary"):
            if mode == "Se connecter":
                u, s, err = auth_login(email, pwd)
            else:
                if pwd != pwd2:
                    st.error("Les mots de passe ne correspondent pas.")
                    st.stop()
                u, s, err = auth_signup(email, pwd)
            if err:
                st.error(err)
            elif u and s:
                st.session_state.user  = {"id":str(u.id),"email":u.email}
                st.session_state.token = s.access_token
                st.rerun()
            else:
                st.success("Compte créé ! Connectez-vous.")

st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyser", "🗺️ Où jeter ?", "📋 Historique"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Analyser
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    source = st.radio("Source", ["📷 Caméra","📁 Upload"], horizontal=True, label_visibility="collapsed")
    image_data = st.camera_input("") if source == "📷 Caméra" else st.file_uploader("", type=["jpg","jpeg","png","webp"])

    if image_data:
        img_id = getattr(image_data, "file_id", id(image_data))
        image  = Image.open(image_data)
        st.image(image, use_container_width=True)

        with st.spinner("🌿 Analyse..."):
            model = load_model()
            if model is None:
                st.error("❌ Modèle non disponible.")
            else:
                image_data.seek(0)
                raw = image_data.read()
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    tensor = TRANSFORM(img).unsqueeze(0)
                    with torch.no_grad():
                        out = model(tensor)
                        prob = torch.softmax(out, dim=1)
                        conf, cid = torch.max(prob, dim=1)
                    cat  = CATEGORIES[cid.item()]
                    conf = round(conf.item()*100, 1)
                    color = BAC_COLORS.get(cat["bac"],"#2d6a4f")
                    txt_color = "#000" if color == "#f5c518" else "white"

                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("Déchet", f"{cat['emoji']} {cat['label']}")
                    conf_color = "#52b788" if conf>=80 else "#f4a261" if conf>=60 else "#e63946"
                    c2.metric("Confiance", f"{conf}%")

                    st.markdown(
                        f"<div style='background:{color};color:{txt_color};padding:16px;border-radius:12px;"
                        f"font-size:18px;font-weight:bold;text-align:center;margin:10px 0'>🗑️ {cat['bac']}</div>",
                        unsafe_allow_html=True)
                    st.info(f"💡 {cat['consigne']}")

                    if conf < 60:
                        st.warning("⚠️ Confiance faible — essayez sur fond neutre.")

                    # Sauvegarde unique
                    if st.session_state.last_img_id != img_id:
                        st.session_state.last_img_id = img_id
                        if st.session_state.user and st.session_state.token:
                            db_save(st.session_state.user["id"], st.session_state.token,
                                    cat["label"], cat["bac"], conf)
                            st.success("✅ Scan sauvegardé.")

                except (UnidentifiedImageError, Exception) as e:
                    st.error(f"❌ Image invalide : {e}")
    else:
        st.markdown("""
        <div style='text-align:center;padding:32px;border:2px dashed #2d6a4f;border-radius:16px;margin-top:12px'>
            <div style='font-size:40px'>📸</div>
            <p style='color:#95d5b2;margin:8px 0'>Prenez une photo ou uploadez une image</p>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Où jeter ?
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    type_dechet = st.selectbox("Type de déchet", list(OU_JETER_TYPES.keys()))
    config = OU_JETER_TYPES[type_dechet]
    st.markdown(f"<div style='background:#1b4332;border-left:4px solid #52b788;border-radius:8px;padding:10px;margin:8px 0'>"
                f"<span style='color:#95d5b2'>💡 {config['conseil']}</span><br>"
                f"<span style='color:#52b788;font-size:12px'>{config['eco_org']}</span></div>",
                unsafe_allow_html=True)

    if SEARCHBOX_AVAILABLE:
        selected = st_searchbox(search_addr, placeholder="Ex: 10 rue de Rivoli, Paris", key="addr_search")
    else:
        selected = None
        raw_addr = st.text_input("Votre adresse", placeholder="Ex: 10 rue de Rivoli, Paris")

    if st.button("🔍 Rechercher", type="primary"):
        lat = lon = label_addr = None
        if SEARCHBOX_AVAILABLE and isinstance(selected, dict):
            lat, lon, label_addr = selected["lat"], selected["lon"], selected["label"]
        elif not SEARCHBOX_AVAILABLE and raw_addr.strip():
            with st.spinner("📍 Géolocalisation..."):
                lat, lon, label_addr = geocode(raw_addr)
        else:
            st.warning("Sélectionnez ou entrez une adresse.")

        if lat:
            st.success(f"📍 **{label_addr}**")
            with st.spinner("🗺️ Recherche..."):
                elements, err = search_overpass(lat, lon, config)
            if err:
                st.error(err)
            else:
                results, seen = [], set()
                for el in elements:
                    el_lat = el.get("lat") or (el.get("center") or {}).get("lat")
                    el_lon = el.get("lon") or (el.get("center") or {}).get("lon")
                    if not (el_lat and el_lon): continue
                    key = (round(el_lat,5), round(el_lon,5))
                    if key in seen: continue
                    seen.add(key)
                    el["_lat"],el["_lon"] = el_lat, el_lon
                    el["_dist"] = haversine(lat,lon,el_lat,el_lon)
                    results.append(el)
                results.sort(key=lambda x: x["_dist"])
                top = results[:10]

                if not top:
                    st.warning(f"😕 Aucun point dans {config['radius']//1000} km.")
                else:
                    st.success(f"✅ {len(results)} point(s) trouvé(s)")

                    if FOLIUM_AVAILABLE:
                        m = folium.Map(location=[lat,lon], zoom_start=13, tiles="CartoDB positron")
                        folium.Marker([lat,lon], popup="📍 Vous",
                                      icon=folium.Icon(color="red",icon="home")).add_to(m)
                        for el in top:
                            tags = el.get("tags",{})
                            name = html_lib.escape(tags.get("name") or tags.get("operator") or "Point de collecte")
                            folium.Marker([el["_lat"],el["_lon"]],
                                          popup=folium.Popup(f"<b>{name}</b><br>📏 {el['_dist']:.1f} km", max_width=180),
                                          icon=folium.Icon(color="green",icon="leaf")).add_to(m)
                        import streamlit.components.v1 as components
                        components.html(m._repr_html_(), height=280)

                    for i,el in enumerate(top[:8],1):
                        tags = el.get("tags",{})
                        name = html_lib.escape(tags.get("name") or tags.get("operator") or "Point de collecte")
                        parts = [tags.get("addr:housenumber",""),tags.get("addr:street",""),tags.get("addr:city","")]
                        addr_str = html_lib.escape(" ".join(p for p in parts if p))
                        st.markdown(
                            f"<div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;"
                            f"padding:12px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px'>"
                            f"<div><b style='color:#d8f3dc'>{i}. {name}</b>"
                            f"{f'<br><span style=\"color:#95d5b2;font-size:12px\">📍 {addr_str}</span>' if addr_str else ''}</div>"
                            f"<span style='background:#2d6a4f;color:#d8f3dc;padding:3px 10px;border-radius:20px;font-size:12px'>📏 {el['_dist']:.1f} km</span></div>",
                            unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Historique
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.user:
        st.markdown("""
        <div style='text-align:center;padding:32px;border:2px dashed #2d6a4f;border-radius:16px'>
            <div style='font-size:36px'>🔒</div>
            <p style='color:#95d5b2'>Connectez-vous pour voir votre historique.</p>
        </div>""", unsafe_allow_html=True)
    else:
        scans = db_get(st.session_state.user["id"], st.session_state.token)
        if not scans:
            st.markdown("""
            <div style='text-align:center;padding:32px;border:2px dashed #2d6a4f;border-radius:16px'>
                <div style='font-size:36px'>🌿</div>
                <p style='color:#95d5b2'>Aucun scan encore.</p>
            </div>""", unsafe_allow_html=True)
        else:
            conf_moy = sum(s["confidence"] for s in scans) / len(scans)
            labels   = [s["label"] for s in scans]
            top_label = max(set(labels), key=labels.count)
            c1,c2,c3 = st.columns(3)
            c1.metric("Scans",len(scans))
            c2.metric("Conf. moy.",f"{conf_moy:.1f}%")
            c3.metric("+ fréquent",top_label)
            st.divider()
            for s in scans:
                color = BAC_COLORS.get(s["bac"],"#2d6a4f")
                conf  = s["confidence"]
                cc    = "#52b788" if conf>=80 else "#f4a261" if conf>=60 else "#e63946"
                heure = s.get("scanned_at","")[:16].replace("T"," ")
                st.markdown(
                    f"<div style='background:#1b4332;border:1px solid #2d6a4f;border-radius:10px;"
                    f"padding:10px;margin:4px 0;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px'>"
                    f"<span style='color:#95d5b2;font-size:12px'>🕐 {heure}</span>"
                    f"<span style='color:#d8f3dc;font-weight:bold'>{s['label']}</span>"
                    f"<span style='background:{color};color:{'#000' if color=='#f5c518' else 'white'};padding:2px 8px;border-radius:20px;font-size:12px'>{s['bac']}</span>"
                    f"<span style='color:{cc};font-weight:bold'>{conf}%</span></div>",
                    unsafe_allow_html=True)
            st.divider()
            if st.button("🗑️ Effacer mon historique", type="secondary"):
                db_clear(st.session_state.user["id"], st.session_state.token)
                st.rerun()
