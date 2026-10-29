FROM python:3.11-slim

WORKDIR /app

COPY backend.requirements.txt .
RUN pip install --no-cache-dir -r backend.requirements.txt
# PyTorch en su build solo-CPU (el contenedor no usa GPU); mismas versiones que requirements-dl.txt
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/raw/ ./data/raw/
# P(O | L) de la red bayesiana: matriz de confusión medida en el test de visión
COPY experiments/vision/confusion_hoja.json ./experiments/vision/confusion_hoja.json
# Los pesos y las tablas de evidencia (models/) NO van en la imagen: se montan como volumen (ver docker-compose.yml)

ENV PYTHONPATH=/app
ENV COFFEEMIND_DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
