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

# Priorité : v3 (EfficientNet) > v2 (MobileNetV3 amélioré) > v1 (baseline)
if (CHECKPOINTS / "waste_ai_v3.pt").exists():
    MODEL_PATH = CHECKPOINTS / "waste_ai_v3.pt"
    MODEL_TYPE = "efficientnet_b2"
elif (CHECKPOINTS / "waste_ai_v2.pt").exists():
    MODEL_PATH = CHECKPOINTS / "waste_ai_v2.pt"
    MODEL_TYPE = "mobilenet_v3"
else:
    MODEL_PATH = CHECKPOINTS / "waste_ai.pt"
    MODEL_TYPE = "mobilenet_v3"

app = FastAPI(title="Waste AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORIES = {
    0: {
        "label": "Carton",
        "bac": "Bac jaune",
        "consigne": "Aplatissez le carton et déposez-le dans le bac jaune.",
    },
    1: {
        "label": "Verre",
        "bac": "Colonne à verre",
        "consigne": "Déposez dans une colonne à verre. Ne mettez pas le couvercle.",
    },
    2: {
        "label": "Métal",
        "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Ecrasez les canettes si possible.",
    },
    3: {
        "label": "Papier",
        "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Pas de papier gras ni de mouchoirs.",
    },
    4: {
        "label": "Plastique",
        "bac": "Bac jaune",
        "consigne": "Déposez dans le bac jaune. Videz et rincez les emballages.",
    },
    5: {
        "label": "Résidus",
        "bac": "Bac gris (ordures ménagères)",
        "consigne": "Déposez dans le bac gris. Cet objet n'est pas recyclable.",
    },
}

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
            m = timm.create_model("efficientnet_b2", pretrained=False, num_classes=len(CATEGORIES))
        else:
            m = models.mobilenet_v3_small(weights=None)
            m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, len(CATEGORIES))

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
