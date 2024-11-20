FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirement.txt

EXPOSE 80

CMD ["fastapi", "run", "main.py","--port", "80"]