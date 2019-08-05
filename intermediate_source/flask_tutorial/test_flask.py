import requests

resp = requests.post("http://localhost:5000/predict", 
                     files={"file": open('/home/ubuntu/03_pytorch-tutorials/_static/img/cat.jpg','rb')})

print(resp.json())