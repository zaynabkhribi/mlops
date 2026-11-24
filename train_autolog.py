
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Charger les données
df = pd.read_csv("churn_data.csv")

# 2. Préparer les variables explicatives
X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

# 3. Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["churn"],
    test_size=0.2,
    random_state=42
)

# 4. Configurer l'expérience MLflow
mlflow.set_experiment("churn-prediction")

# 5. Activer l'enregistrement automatique pour Scikit-learn
# MLflow enregistre automatiquement :
# - Les paramètres du modèle
# - Les métriques
# - Le modèle lui-même

mlflow.sklearn.autolog(
    serialization_format="cloudpickle"
)

# 6. Entraîner le modèle
with mlflow.start_run(run_name="essai-auto"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)  # L'enregistrement se déclenche au fit()

    # 7. Évaluer le modèle sur les données de test
    # score() calcule l'accuracy
    print("Score test :", model.score(X_test, y_test))
