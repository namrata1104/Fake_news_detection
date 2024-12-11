FROM python:3.10-slim

#WORKDIR /fake_news_detection

#COPY fake_news_detection fake_news_detection
#COPY requirements.txt requirements.txt

COPY . /fake_news_detection
WORKDIR /fake_news_detection

RUN pip install --upgrade pip && pip install -r requirements.txt

#EXPOSE 5000

CMD uvicorn fake_news_detection.api.fast:app --host 0.0.0.0 --port $PORT
