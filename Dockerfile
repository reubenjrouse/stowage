FROM python:3.11-slim

WORKDIR /app

# The CPU wheel is ~500MB; the CUDA one is ~2.5GB and useless here -- the
# policy is 71k parameters and inference runs on CPU.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# torch sizes its thread pools to the HOST's core count, not our CPU
# allocation, and the per-thread arenas alone pushed the container past 1GiB
# before it could bind. These must be set before torch is imported.
ENV OMP_NUM_THREADS=1     MKL_NUM_THREADS=1     OPENBLAS_NUM_THREADS=1

# so print() and tracebacks reach Cloud Run's logs immediately
ENV PYTHONUNBUFFERED=1

# Cloud Run overrides this with its own PORT; the default is for running the
# image locally.
ENV PORT=8080
EXPOSE 8080

CMD ["python", "backend/app.py"]
