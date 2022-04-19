# P7-project

Déploiment du dashboard Plotly sur Heroku
Le projet est découpé en 4 scripts : 
P7_script_1  :  EDA et obtention d'une base de données unifiées, aggrégées et nettoyée
P7_script_2 : Elaboration et entraînement du modèle
P7_script_3 : Utilisation du modèle pour prédire la probabilité d'une observation d'appartenir à une classe (+ réalisation de graphiques). Ce programme est appelé par le programme P7_script_4 et lui renvoie le résultat du predict_proba du modèle
P7_script_4 : Réalisation du dashboard (appel du programme P7_script_3 pour la réalisation de la prédiction)
