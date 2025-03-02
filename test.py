import requests

url = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "format": "json",
    "titles": "Gemini (chatbot)",
    "prop": "extracts",
    "exintro": True,
    "explaintext": True
}

response = requests.get(url, params=params)
print(response.json())
