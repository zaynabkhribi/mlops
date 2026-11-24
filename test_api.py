import requests # envoyer des requêtes HTTP depuis Python
URL = "http://localhost:8000" # l'adresse de notre API
# 1. L'API est-elle en vie ?
print("GET /health ->", requests.get(URL + "/health").json())
# 2. Un client à faible consommation et faible ancienneté
client = {"age": 35, "salaire": 2500, "abonnement": "Standard",
"anciennete": 1, "consommation": 10}
r = requests.post(URL + "/predict", json=client) # json= : envoie le dictionnaire en JSON
print("POST /predict (client 1) ->", r.status_code, r.json())
# 3. Un client fidèle : forte consommation, forte ancienneté
client2 = {"age": 50, "salaire": 4000, "abonnement": "Premium",
"anciennete": 10, "consommation": 80}
r = requests.post(URL + "/predict", json=client2)
print("POST /predict (client 2) ->", r.status_code, r.json())
# 4. Une erreur volontaire : l'âge est du texte
mauvais = dict(client, age="abc") # copie de client avec age modifié
r = requests.post(URL + "/predict", json=mauvais)
print("POST /predict (age='abc') ->", r.status_code)
print("Message :", r.json()["detail"][0]["msg"])