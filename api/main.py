from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from pathlib import Path

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from torchvision import models

CHECKPOINTS = Path(__file__).parent.parent / "model" / "checkpoints"

# Priorité : v6 (8 classes) > v4/v3 (6 classes) > v2/v1
if (CHECKPOINTS / "waste_ai_v6.pt").exists():
    MODEL_PATH = CHECKPOINTS / "waste_ai_v6.pt"
    MODEL_TYPE = "efficientnet_b2"
    NUM_CLASSES = 8
elif (CHECKPOINTS / "waste_ai_v4.pt").exists():
    MODEL_PATH = CHECKPOINTS / "waste_ai_v4.pt"
    MODEL_TYPE = "efficientnet_b2"
    NUM_CLASSES = 6
elif (CHECKPOINTS / "waste_ai_v3.pt").exists():
    MODEL_PATH = CHECKPOINTS / "waste_ai_v3.pt"
    MODEL_TYPE = "efficientnet_b2"
    NUM_CLASSES = 6
elif (CHECKPOINTS / "waste_ai_v2.pt").exists():
    MODEL_PATH = CHECKPOINTS / "waste_ai_v2.pt"
    MODEL_TYPE = "mobilenet_v3"
    NUM_CLASSES = 6
else:
    MODEL_PATH = CHECKPOINTS / "waste_ai.pt"
    MODEL_TYPE = "mobilenet_v3"
    NUM_CLASSES = 6

app = FastAPI(title="Waste AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 10 classes (ordre alphabétique, utilisé par le modèle v5)
# battery=0, bulb=1, cardboard=2, electronic=3, glass=4,
# medicine=5, metal=6, paper=7, plastic=8, trash=9
CATEGORIES_V5 = {
    0: {"label": "Piles & Batteries", "bac": "Bac à piles (magasin / bureau de tabac)",
        "consigne": "Ne jetez jamais une pile à la poubelle. Déposez-la dans un bac de collecte en supermarché, bureau de tabac ou magasin d'électronique."},
    1: {"label": "Ampoules",          "bac": "Point de collecte (magasin bricolage / Ikea)",
        "consigne": "Déposez dans un point de collecte Ecosystem. Ne mettez pas au bac jaune ni à la poubelle ordinaire."},
    2: {"label": "Carton",            "bac": "Bac jaune",
        "consigne": "Aplatissez le carton et déposez-le dans le bac jaune."},
    3: {"label": "Électronique",      "bac": "Déchetterie (point DEEE)",
        "consigne": "Les appareils électroniques vont en déchetterie (zone DEEE). Le vendeur est obligé de reprendre votre ancien appareil lors d'un achat (loi AGEC)."},
    4: {"label": "Verre",             "bac": "Colonne à verre",
        "consigne": "Déposez dans une colonne à verre. Retirez les couvercles. Ne mettez pas les vitres ni la vaisselle."},
    5: {"label": "Médicaments",       "bac": "Pharmacie (réseau Cyclamed)",
        "consigne": "Rapportez vos médicaments non utilisés ou périmés en pharmacie. Ne les jetez jamais à la poubelle ou dans les toilettes."},
    6: {"label": "Métal",             "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Écrasez les canettes si possible."},
    7: {"label": "Papier",            "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Pas de papier gras ni de mouchoirs."},
    8: {"label": "Plastique",         "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Videz et rincez les emballages."},
    9: {"label": "Résidus",           "bac": "Bac gris (ordures ménagères)",
        "consigne": "Déposez dans le bac gris. Cet objet n'est pas recyclable."},
}

# 6 classes (ordre alphabétique ImageFolder pour v3/v4)
# cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5
CATEGORIES_V6 = {
    0: {"label": "Carton",    "bac": "Bac jaune",                   "consigne": "Aplatissez le carton et déposez-le dans le bac jaune."},
    1: {"label": "Verre",     "bac": "Colonne à verre",             "consigne": "Déposez dans une colonne à verre. Ne mettez pas le couvercle."},
    2: {"label": "Métal",     "bac": "Bac jaune",                   "consigne": "Déposez dans le bac jaune. Ecrasez les canettes si possible."},
    3: {"label": "Papier",    "bac": "Bac jaune",                   "consigne": "Déposez dans le bac jaune. Pas de papier gras ni de mouchoirs."},
    4: {"label": "Plastique", "bac": "Bac jaune",                   "consigne": "Déposez dans le bac jaune. Videz et rincez les emballages."},
    5: {"label": "Résidus",   "bac": "Bac gris (ordures ménagères)", "consigne": "Déposez dans le bac gris. Cet objet n'est pas recyclable."},
}

# 8 classes v6 (ordre alphabétique ImageFolder)
# bulb=0, cardboard=1, electronic=2, glass=3, metal=4, paper=5, plastic=6, trash=7
CATEGORIES_V8 = {
    0: {"label": "Ampoules",     "bac": "Point de collecte (magasin bricolage / Ikea)",
        "consigne": "Ne jetez pas à la poubelle. Déposez dans un point de collecte Ecosystem en magasin."},
    1: {"label": "Carton",       "bac": "Bac jaune",
        "consigne": "Aplatissez le carton et déposez-le dans le bac jaune."},
    2: {"label": "Électronique", "bac": "Déchetterie (point DEEE)",
        "consigne": "Déposez en déchetterie (zone DEEE). Le vendeur doit reprendre votre ancien appareil lors d'un achat (loi AGEC)."},
    3: {"label": "Verre",        "bac": "Colonne à verre",
        "consigne": "Déposez dans une colonne à verre. Retirez les couvercles."},
    4: {"label": "Métal",        "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Écrasez les canettes si possible."},
    5: {"label": "Papier",       "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Pas de papier gras ni de mouchoirs."},
    6: {"label": "Plastique",    "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Videz et rincez les emballages."},
    7: {"label": "Résidus",      "bac": "Bac gris (ordures ménagères)",
        "consigne": "Déposez dans le bac gris. Cet objet n'est pas recyclable."},
}

if NUM_CLASSES == 8:
    CATEGORIES = CATEGORIES_V8
elif NUM_CLASSES == 10:
    CATEGORIES = CATEGORIES_V5
else:
    CATEGORIES = CATEGORIES_V6

# EfficientNet-B2 utilise des images 260x260
IMG_SIZE = 260 if MODEL_TYPE == "efficientnet_b2" else 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = None


def load_model():
    global model
    try:
        if MODEL_TYPE == "efficientnet_b2" and TIMM_AVAILABLE:
            import timm
            m = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
        else:
            m = models.mobilenet_v3_small(weights=None)
            m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, NUM_CLASSES)

        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        model = m
        print(f"Modele charge : {MODEL_TYPE} depuis {MODEL_PATH.name}")
    except FileNotFoundError:
        print("Aucun checkpoint trouve.")
    except Exception as e:
        print(f"Erreur chargement modele : {e}")


load_model()


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_TYPE,
        "checkpoint": MODEL_PATH.name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non disponible.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier invalide. Envoyez une image.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

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
    }
