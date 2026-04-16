# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt \
#     && pip install --no-cache-dir scikit-learn

# COPY src/ src/
# COPY params.yaml .
# COPY entrypoint.sh .
# RUN chmod +x entrypoint.sh

# EXPOSE 3000

# CMD ["./entrypoint.sh"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scikit-learn

COPY src/ src/
COPY params.yaml .
COPY entrypoint.sh .

# Fix Windows CRLF line endings — harmless on Linux, essential on Windows hosts
RUN sed -i 's/\r//' entrypoint.sh && chmod +x entrypoint.sh

EXPOSE 3000

CMD ["./entrypoint.sh"]