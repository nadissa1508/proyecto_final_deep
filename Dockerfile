FROM python:3.11-slim

WORKDIR /app

COPY backend.requirements.txt .
RUN pip install --no-cache-dir -r backend.requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/

ENV PYTHONPATH=/app
ENV COFFEEMIND_DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
