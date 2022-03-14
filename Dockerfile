FROM python:3.8.12-buster

COPY api /api
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY resources /resources

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.vibe:app --host 0.0.0.0 --port $PORT

# CMD uvicorn api.vibe:app --host 0.0.0.0

# CMD uvicorn api.vibe:app --host 0.0.0.0 --port $PORT
