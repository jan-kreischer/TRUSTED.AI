# === DOCKER COMMANDS ===

1. Run command to build Docker Image
docker build -f ./Dockerfile -t trustedai/webapp .

2. List all Docker Images
docker image ls


3. Run the docker image as a docker container 
docker run trustedai/webapp -p 8080:8080

docker container ls
