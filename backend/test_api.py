import requests

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "text": "I feel very nervous and worried"
    }
)

print(response.json())