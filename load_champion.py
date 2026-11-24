
import mlflow
import mlflow.sklearn
import pandas as pd

# Utiliser la même base SQLite
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/GFI/mlops/tp1/mlflow.db"
)

# Charger le modèle champion
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

print("Modèle chargé :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))

# Nouveau client : adapte les valeurs à ton jeu de données
client = pd.DataFrame([{
    "age": 35,
    "abonnement": 1,
    "anciennete": 1,
    "consommation": 10
}])

# Respecter les variables et leur ordre
client = client[list(model.feature_names_in_)]

prediction = model.predict(client)[0]

print(
    "Le client va-t-il partir ? ->",
    "OUI" if prediction == 1 else "NON"
)
