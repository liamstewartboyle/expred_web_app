FROM huggingface/transformers-tensorflow-cpu

ENV PYTHONBUFFERED True

ENV PORT 8080
ENV HOST 0.0.0.0

ENV APP_HOME /faxplain
WORKDIR $APP_HOME
COPY faxplain/ /faxplain

RUN apt-get update
RUN apt-get install -y libc-dev curl build-essential gunicorn3
RUN apt-get install -y locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD exec gunicorn -b $HOST:$PORT -w 1 --thread 8 --timeout 0 faxplain:app
