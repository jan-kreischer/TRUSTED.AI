# === DOCKER ===
this file contains a small explanation
on how to use Docker for our TrustedAI project.

1. Go to the /webapp folder of our project

2. Run docker build command in order to build the Docker Image from a Dockerfile
- docker build -f <path-to-dockerfile> -t <hub-user>/<repo-name> .
- docker build -f ./Dockerfile -t trustedai/webapp:v1 .
-f ... specify path to Dockerfile
-t ... add a tag to the image

3. List all locally existing Docker images
- docker image ls 
- docker images

4. Run the docker image as a local docker container 
and check if everything is working correctly
- docker run -p <host-port>:<container-port> trustedai/webapp:v1
- docker run -p 8080:8080 trustedai/webapp:v1
   
5. Login to Dockerhub
- docker login --username trustedai
    
6. Push the finished Docker Images to Dockerhub
- docker push <hub-user>/<repo-name>:<tag>
- docker push trustedai/webapp:v1

    
7. Run Docker compose
docker compose -f docker-compose.yml up

docker network create -d bridge proxynet
    
# === DOCKER COMMANDS ===


