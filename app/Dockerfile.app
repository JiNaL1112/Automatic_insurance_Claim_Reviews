# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY src/ src/

# WORKDIR /app/src/api

# EXPOSE 5005

# CMD ["python", "flask_app.py"]


FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Set PYTHONPATH so 'from api.config import ...' resolves correctly
ENV PYTHONPATH=/app/src

EXPOSE 5005

CMD ["python", "src/api/flask_app.py"]
