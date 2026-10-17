FROM python:3.11-slim

WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

EXPOSE 8000

CMD ["sh", "-c", "python seed_graph.py && uvicorn main:app --host 0.0.0.0 --port 8000"]