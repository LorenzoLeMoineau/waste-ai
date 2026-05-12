# Waste AI

> Prototype d'assistance au tri des déchets par vision artificielle

**Projet transverse — Semestre 2026 | École d'ingénieurs informatique**  
Équipe : Nicolas · Jeremy · Arthur · Thomas · Louis · Lorenzo

---

## Objectif

Concevoir un prototype capable d'**identifier une catégorie de déchet** à partir d'une image et de proposer une **orientation de tri compréhensible**, alignée sur les standards ADEME / Citeo.

## Fonctionnement

```
[Photo utilisateur] -> [Modele CNN (MobileNetV3)] -> [Categorie + Consigne + Confiance]
```

1. **Capture** — L'utilisateur prend une photo ou uploade une image
2. **Analyse** — Le modèle classifie le déchet parmi les catégories couvertes
3. **Recommandation** — L'app affiche la catégorie et la consigne de tri
4. **Confiance** — Un score de confiance et une explication courte accompagnent la décision

## Catégories couvertes

| Catégorie | Bac |
|-----------|-----|
| Plastique | Bac jaune (recyclage) |
| Papier / Carton | Bac jaune (recyclage) |
| Métal | Bac jaune (recyclage) |
| Verre | Colonne à verre (vert) |
| Organique | Bac compost (marron) |
| Résidus (tout-venant) | Bac ordures (gris) |

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Modèle | PyTorch + MobileNetV3 (transfer learning) |
| Entraînement | Google Colab |
| Backend | FastAPI (Python) |
| Frontend démo | Streamlit |

## Structure du projet

```
waste-ai/
├── data/
│   ├── raw/          # Datasets bruts (TrashNet, TACO...)
│   └── processed/    # Images nettoyées et annotées
├── model/
│   └── checkpoints/  # Poids du modèle entraîné (.pt)
├── notebooks/        # Notebooks Colab d'entraînement
├── api/              # Backend FastAPI
├── frontend/         # Interface Streamlit
└── requirements.txt
```

## Datasets

- [TrashNet](https://github.com/garythung/trashnet) — référence académique, ~2500 images, 6 classes
- [TACO](http://tacodataset.org/) — photos en contexte réel
- Captures terrain de l'équipe

## Références officielles

- [ADEME](https://www.ademe.fr/) — consignes de tri en France
- [Citeo](https://www.citeo.com/) — référentiel emballages ménagers

## Installation

```bash
pip install -r requirements.txt

# Backend
cd api && uvicorn main:app --reload

# Frontend
cd frontend && streamlit run app.py
```
