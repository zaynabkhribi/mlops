# 1. Image de base : Linux minimal avec Python 3.11 déjà installé
FROM python:3.11-slim

# 2. Dossier de travail dans l'image (créé s'il n'existe pas)
WORKDIR /app

# 3. Copier d'abord uniquement la liste des bibliothèques, puis les installer.
# Cette couche est mise en cache : tant que requirements.txt ne change pas,
# Docker ne refait pas l'installation (gain de plusieurs minutes).
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. Copier ensuite le code et le modèle (ils changent plus souvent)
COPY app.py .
COPY model/ model/

# 5. Valeur par défaut : charger le modèle depuis le dossier model
ENV MODEL_URI=model

# 6. Documenter le port utilisé par l'API
# (cela fournit uniquement une information et ne publie pas le port)
EXPOSE 8000

# 7. Commande lancée au démarrage du conteneur.
# --host 0.0.0.0 : écouter sur toutes les interfaces réseau du conteneur
# (avec 127.0.0.1, l'API serait inaccessible depuis l'extérieur)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]