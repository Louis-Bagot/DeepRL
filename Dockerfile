FROM naoshikuu/base_containers:deeprl
#FROM naoshikuu/base_containers:torch2
MAINTAINER louis.bagot@uantwerpen.be
COPY . /app
WORKDIR /app
ENTRYPOINT python3 main.py
#RUN python main.py
