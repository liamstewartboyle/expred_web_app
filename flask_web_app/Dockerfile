FROM huggingface/transformers-pytorch-cpu

ENV PYTHONBUFFERED True
ENV PYTHON python3.6

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY app/ /app

RUN apt-get update
RUN apt-get install -y apt-utils
#RUN apt-get install -y libc-dev curl build-essential python-pip gunicorn3 locales locales-all
# RUN apt-get remove -y python3.6
# RUN apt-get install -y python3.8 python3-pip python3-dev pypy-dev 
RUN apt-get install -y gunicorn3 locales locales-all
# RUN apt-get install -y locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
RUN $PYTHON -m pip install --upgrade pip
# RUN $PYTHON -m pip install -e git+https://github.com/pypa/pip.git@22.1.1#egg=pip
# RUN $PYTHON -m pip install git+https://github.com/benoitc/gunicorn.git
# RUN $PYTHON -m pip install Cython
RUN $PYTHON -m pip install -r requirements.txt
RUN mkdir -p .keras/trained_models/fever
RUN mkdir -p .keras/trained_models/movies
RUN mkdir -p .keras/datasets/movies
RUN mkdir -p .keras/datasets/fever

ENV PORT 8080
# ENV PORT 80
ENV HOST 0.0.0.0
CMD exec gunicorn3 -b $HOST:$PORT -w 1 --thread 8 --timeout 0 main:app
#CMD exec python3 -m gunicorn -b $HOST:$PORT -w 1 --thread 8 --timeout 0 faxplain:ap
