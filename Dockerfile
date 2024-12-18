FROM python:3.10-slim

#WORKDIR /fake_news_detection

COPY data data
COPY fake_news_detection fake_news_detection
COPY training_outputs training_outputs
COPY .env .env
COPY requirements.txt requirements.txt


COPY . /fake_news_detection
WORKDIR /fake_news_detection

RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

EXPOSE 5000

CMD uvicorn fake_news_detection.api.fast:app --host 0.0.0.0 --port $PORT
