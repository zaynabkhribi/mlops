import pandas as pd # pandas : pour construire et manipuler notre tableau de données
import numpy as np # numpy : pour générer des valeurs numériques aléatoires

np.random.seed(42)
n = 500 # nombre de clients fictifs à générer
# On construit le tableau colonne par colonne.
# Chaque colonne ci-dessous est une FEATURE (donnée d'entrée du futur modèle).
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n), # âge entre 18 et 70 ans
    "salaire": np.random.randint(1200, 8000, n), # salaire mensuel simulé
    "abonnement": np.random.choice(["Basique", "Standard", "Premium"], n), # type d'offre
    "anciennete": np.random.randint(0, 15, n), # ancienneté en années
    "consommation": np.random.randint(0, 100, n), # score d'usage du service, sur 100
    })

data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int) # True/False est converti en 1/0
# On sauvegarde le tableau dans un fichier CSV, sur le disque.
# ATTENTION : ce fichier n'est protégé par aucun système de version à ce stade.
data.to_csv("churn_data.csv", index=False)
# Deux affichages de contrôle, juste pour vérifier visuellement le résultat.
print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")