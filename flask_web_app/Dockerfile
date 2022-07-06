FROM huggingface/transformers-pytorch-cpu

ENV PYTHONBUFFERED True
ENV PYTHON python3.6

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY app/ /app

RUN apt-get update
RUN apt-get install -y apt-utils gunicorn3 locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
RUN $PYTHON -m pip install --upgrade pip
RUN $PYTHON -m pip install -r requirements.txt
RUN mkdir -p .keras/trained_models/fever
RUN mkdir -p .keras/trained_models/movies
RUN mkdir -p .keras/datasets/movies
RUN mkdir -p .keras/datasets/fever

ENV PORT 8080
ENV HOST 0.0.0.0
CMD exec gunicorn3 -b $HOST:$PORT -w 1 --thread 8 --timeout 0 main:app
