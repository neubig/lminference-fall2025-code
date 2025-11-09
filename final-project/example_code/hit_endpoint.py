import requests

# Modal will give you a URL after deployment, you will need to edit yourModalID to your modal username
url = "https://yourModalID--andrewid-1-model-completions.modal.run"

response = requests.post(
    url,
    json={
        "prompt": ["Once upon a time", "testing: "],
    }
)
print(response)
print(response.json()["choices"][0]["text"])

print(response.json())


response = requests.post(
    url,
    json={
        "prompt": "single-example inference also works! see,",
    }
)
print(response)
print(response.json()["choices"][0]["text"])

print(response.json())

