FROM python:3.9-slim

ENV PYTHONBUFFERED True

ENV PORT 8080
ENV HOST 127.0.0.1

ENV APP_HOME /faxplain
WORKDIR $APP_HOME
COPY faxplain/ /faxplain

RUN apt-get update
RUN apt-get install -y libc-dev
RUN apt-get install -y build-essential
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD exec gunicorn -b $HOST:$PORT -w 1 --thread 8 --timeout 0 faxplain:app
