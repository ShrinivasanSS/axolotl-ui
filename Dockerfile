FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#COPY . .

ENV FLASK_APP=wsgi.py
ENV FLASK_ENV=production
ENV DATABASE_URL=sqlite:////app/instance/app.db

RUN mkdir -p /app/instance && mkdir -p /app/data/datasets /app/data/configs /app/data/logs

EXPOSE 5000

CMD ["gunicorn", "wsgi:app", "-b", "0.0.0.0:5000", "--timeout", "600"]
