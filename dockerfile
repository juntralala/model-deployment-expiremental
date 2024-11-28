FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirement.txt

EXPOSE 80

CMD ["fastapi", "run", "main.py","--port", "80"]