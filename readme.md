# create docker image
sudo docker build -t bge-embedding-service .


sudo docker run -p 8000:8000 bge-embedding-service

sudo docker run -p 8000:8000 bge-embedding-service
#

uses bge-small-en model from huggingface

curl -X POST http://localhost:8000/embed -H "Content-Type: application/json" -d '{"texts": ["hello world", "this is a test"]}'
