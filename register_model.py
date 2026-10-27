# ===== 1. Importer les bibliothèques =====
import mlflow # le module principal de MLflow
from mlflow import MlflowClient # le « client » : permet de gérer le Registry
# ===== 2. Se connecter à la même base que le TP3 =====
# C'est le fichier mlflow.db du dossier courant, qui contient tous nos runs
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Retrouver le meilleur run (même logique qu'au TP3) =====
runs = mlflow.search_runs(
experiment_names=["churn-prediction"],
order_by=["metrics.f1 DESC"]) # f1 décroissant : le meilleur en premier
best = runs.iloc[0] # première ligne = meilleur run
run_id = best["run_id"] # son identifiant unique
print("Meilleur run :", best["tags.mlflow.runName"], "| id :", run_id[:8], "...")

# ===== 4. Enregistrer le modèle de ce run dans le Registry =====

result = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("Modele enregistre : churn-model | version", result.version)
# ===== 5. Poser l'alias « champion » sur cette version =====
client = MlflowClient()
client.set_registered_model_alias("churn-model", "champion", result.version)
print("Alias 'champion' -> version", result.version)