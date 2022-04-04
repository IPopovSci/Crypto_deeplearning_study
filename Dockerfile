FROM  continuumio/miniconda3
LABEL Author, Ivan Popov

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

#---------------- Prepare the envirennment
RUN conda update --name base conda &&\
    conda env create --name env --file environment.yml

SHELL ["conda", "run", "--name", "env", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--name", "env","--no-capture-output", "python","-u", "Main.py"]