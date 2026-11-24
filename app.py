# ===== 1. Importer les bibliothèques =====
import os  # Lire les variables d'environnement
from typing import Literal  # Limiter les valeurs possibles d'un champ

import mlflow  # Charger le modèle (Registry ou dossier)
import mlflow.sklearn
import pandas as pd  # Fabriquer le tableau attendu par le modèle
from fastapi import FastAPI  # Le framework de l'API
from pydantic import BaseModel  # Décrire et valider les données reçues


# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")  # Le titre s'affiche dans la page /docs


# ===== 3. Lire la configuration dans les variables d'environnement =====
# Rien n'est écrit « en dur » : le même code marche sur le poste,
# dans un conteneur seul ou dans Docker Compose.
# Seules les variables changent.
#
# os.getenv("NOM", "valeur") :
# renvoie la valeur de la variable ou la valeur par défaut si elle n'existe pas.

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")


# ===== 4. Charger le champion UNE SEULE FOIS, au démarrage =====
# MODEL_URI peut être une adresse du Registry (models:/...)
# ou un simple dossier contenant le modèle.

mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Modèle chargé depuis : {MODEL_URI}")


# ===== 5. Décrire les données attendues (validées automatiquement) =====
class Client(BaseModel):
    age: int  # Nombre entier
    salaire: float  # Nombre décimal
    abonnement: Literal["Basique", "Standard", "Premium"]  # 3 valeurs possibles
    anciennete: float
    consommation: float


# ===== 6. Route /health : vérifier que l'API est en vie =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 7. Route /predict : recevoir un client et renvoyer la prédiction =====
@app.post("/predict")
def predict(client: Client):
    # a) Transformer le client reçu en tableau d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer le texte « abonnement » en colonnes 0/1,
    # comme lors de l'entraînement
    ligne = pd.get_dummies(ligne)

    # c) Garder les colonnes du modèle dans le bon ordre ;
    # les colonnes absentes prennent la valeur 0
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Prédire : 1 = part, 0 = reste
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        ),
    }
