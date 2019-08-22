FROM jupyter/scipy-notebook

USER root
RUN chmod 777 /home/jovyan/work

USER jovyan
COPY requirements.txt .
RUN nflgame-update-players
RUN pip install -r requirements.txt
