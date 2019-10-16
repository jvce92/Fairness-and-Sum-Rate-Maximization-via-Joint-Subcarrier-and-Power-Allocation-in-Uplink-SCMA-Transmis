FROM python:3.7.4-slim

RUN apt-get update &&\
    apt-get upgrade -y &&\ 
    apt-get install gcc -y &&\
    apt-get install g++ -y &&\ 
    apt-get install git -y 

RUN cd ./home && git clone https://github.com/jvce92/Fairness-and-Sum-Rate-Maximization-via-Joint-Subcarrier-and-Power-Allocation-in-Uplink-SCMA-Transmis

WORKDIR ./home/Fairness-and-Sum-Rate-Maximization-via-Joint-Subcarrier-and-Power-Allocation-in-Uplink-SCMA-Transmis

RUN pip install virtualenv

RUN python --version

RUN virtualenv --python=/usr/local/bin/python env

RUN /bin/bash -c "source ./env/bin/activate"

RUN pip install -r requirements.txt