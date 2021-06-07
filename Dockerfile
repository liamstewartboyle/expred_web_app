FROM python:3.9-slim

ENV PYTHONBUFFERED True

ENV PORT 8080

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install Flask gunicorn

CMD exec gunicorn --bind $PORT --workers 1 --thread8 --timeout 0 main:app
