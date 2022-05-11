import requests

REQUEST_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

if __name__ == "__main__":
    url = "https://api.github.com/repos/pytorch/pytorch/contents/.circleci"

    response = requests.get(url, headers=REQUEST_HEADERS)
    for file in response.json():
        if file["name"] == "docker":
            print(file["sha"])
