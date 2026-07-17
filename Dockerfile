# StorageIQ dashboard — serves precomputed analytics, no ML deps at runtime.
# Run the pipeline first (locally or in CI) so StorageIQ_Dashboard/dashboard_data.json exists:
#   python data/download_backblaze.py --quarter Q1_2025
#   python StorageIQ_Pipeline.py
#   cp outputs/dashboard_data.json StorageIQ_Dashboard/dashboard_data.json
#
# Build & run:
#   docker build -t storageiq .
#   docker run -p 8000:8000 storageiq

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY StorageIQ_Dashboard/ StorageIQ_Dashboard/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--app-dir", "StorageIQ_Dashboard", "--host", "0.0.0.0", "--port", "8000"]
