FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENV PORT=8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 900