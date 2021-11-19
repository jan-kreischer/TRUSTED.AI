
# TRUSTED.AI



## Time Schedule

- [x] Analyze the current state of the art regarding thrustworithniess. (Februar - March)
- [x] Identify missing aspects relevant for automatically calculating a model's trust score (April)
- [x] Create a taxonomy containing relevant pillars and metrics (May)
- [x] Select and analyze a suitable application scenario (June)
- [x] Design a trusted AI algorithm (July)
- [x] Implementation (August - September)
- [x] Evaluation and Discussion (October)
- [x] Documentation and Process (November)



## Webapp

The webapp is build using Flask, Dash and is optimized to run on python version 3.9.7.
In order to setup and run the webapplication you need to execute the following steps.
At first open the console and navigate to the webapp directory.
Then install the necessary dependencies form the requirements.txt file.
Start the application.

```
cd webapp
pip install -r requirements.txt
python index.py
```

## Docker 
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

# Open shell session for container
docker exec -it <container name> /bin/bash



## Documentation 

The whole documentation of the project can be found on our [GitHub page](https://joelleupp.github.io/Trusted-AI/)
The documentation is created with mkdocs, which uses simple markdown file and a .yml config file to build and deploy the documentation to GitHub.

The relevant files to create the documentation are in the [mkdocs folder](https://github.com/JoelLeupp/Trusted-AI/tree/main/mkdocs)

The file mkdocs.yml is the config file and has to be adapted if the theme should change or the index changes or new markdowns added. 
The markdown files are in the folder docs and include the whole documentation.

installation

```sh
pip install --upgrade pip
pip install mkdocs
pip install pymdown-extensions
pip install python-markdown-math
```

workflow:

1.  At the location of the mkdocs folder open the terminal and enter:

        
        mkdocs serve
        

    This will build the documentation and host it locally, where the full documentation can be seen.

2.  change the config or markdown files. Sinc you are serving the documentation locally you can see the changes in the config or markdown files in real time.
You will always see the output and how the documentation will look in the end and therefore minimalize syntax errors in markdown for example. 

3.  deploy the build on GitHub to the branche gh-pages. For this open another termianl at the localtion of the mkdocs folder and enter:
        
        mkdocs gh-deploy
        
4. commit and push the changed config or markdown files (keep the source of the build up to date)
