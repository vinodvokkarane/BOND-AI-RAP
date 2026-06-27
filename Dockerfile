FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ONLY_BINARY=:all:
ENV PIP_PREFER_BINARY=1
ENV PORT=7860

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --only-binary=:all: -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
