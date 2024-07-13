import requests

REQUEST_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

if __name__ == "__main__":
    url = "https://api.github.com/repos/pytorch/pytorch/contents/.ci"

    response = requests.get(url, headers=REQUEST_HEADERS)
    docker_sha = None
    for finfo in response.json():
        if finfo["name"] == "docker":
            docker_sha = finfo["sha"]
            break
    if docker_sha is None:
        raise RuntimeError("Can't find sha sum of docker folder")
    print(docker_sha)
