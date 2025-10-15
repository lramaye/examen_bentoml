import requests

# The URL of the login and prediction endpoints
login_url = "http://127.0.0.1:3001/login"
predict_url = "http://127.0.0.1:3001/v1/models/rf_regressor/predict"

# Données de connexion
credentials = {
    "username": "user123",
    "password": "password123"
}

# Send a POST request to the login endpoint
login_response = requests.post(
    login_url,
    headers={"Content-Type": "application/json"},
    json=credentials
)

# Check if the login was successful
if login_response.status_code == 200:
    token = login_response.json().get("token")
    print("Token JWT obtenu:", token)

    # Data to be sent to the prediction endpoint
    data = {
        "GRE_score": 311,
        "TOEFL_score": 105,
        "University_Rating": 2,
        "SOP": 3.0,
        "LOR": 2.0,
        "CGPA": 8.12,
        "Research": 1
    }

    # Send a POST request to the prediction
    response = requests.post(
        predict_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json=data
    )

    print("Réponse de l'API de prédiction:", response.text)
else:
    print("Erreur lors de la connexion:", login_response.text)