FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scikit-learn

COPY src/ src/
COPY params.yaml .

EXPOSE 3000

CMD ["bentoml", "serve", "src/serving/service.py", "--host", "0.0.0.0", "--port", "3000"]