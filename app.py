import os
import requests
from flask import Flask, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Configurazione (Usa la tua API Key di Football-Data.org)
API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://api.football-data.org/v4/"

def get_live_data(limit=100):
    headers = {'X-Auth-Token': API_KEY}
    # Esempio: recupera le ultime 100 partite della Serie A (ID: SA)
    response = requests.get(f"{BASE_URL}competitions/SA/matches?status=FINISHED", headers=headers)
    data = response.json()
    
    rows = []
    for match in data['matches']:
        rows.append({
            'home_team': match['homeTeam']['name'],
            'away_team': match['awayTeam']['name'],
            'home_score': match['score']['fullTime']['home'],
            'away_score': match['score']['fullTime']['away'],
            'winner': 1 if match['score']['winner'] == 'HOME_TEAM' else (2 if match['score']['winner'] == 'AWAY_TEAM' else 0)
        })
    return pd.DataFrame(rows)

@app.route('/predict/<home_team>/<away_team>')
def predict(home_team, away_team):
    df = get_live_data()
    
    # Feature Engineering Semplice: Media gol passata
    # Nota: In un modello reale, qui calcoleresti le medie storiche dei team
    X = df[['home_score', 'away_score']] # Esempio semplificato
    y = df['winner']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Simuliamo dati di input per la partita attuale (es. medie gol delle ultime 5)
    # Qui dovresti passare le statistiche reali dei due team cercati
    prediction = model.predict([[1.5, 1.2]]) # Esempio di medie gol
    
    res = "Vittoria Casa" if prediction[0] == 1 else ("Vittoria Trasferta" if prediction[0] == 2 else "Pareggio")
    
    return jsonify({
        "match": f"{home_team} vs {away_team}",
        "prediction": res,
        "disclaimer": "L'IA suggerisce in base ai dati storici, non è una certezza."
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
