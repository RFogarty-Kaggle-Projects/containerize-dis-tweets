#The image running on flask/gunicorn
docker build -f docker/flask_a/Dockerfile -t dtweets_flask .

#The image running on fastapi/uvicorn
docker build -f docker/fast_api_a/Dockerfile -t dtweets_fastapi .

