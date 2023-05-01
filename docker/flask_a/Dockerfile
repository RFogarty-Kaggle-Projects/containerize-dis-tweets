
FROM python:3.10-slim-bullseye as base

#Training layer - we train the model within the container then copy over only the files we need for the final image
FROM base as train_model
WORKDIR /build_model
COPY build_model/*.py .
COPY build_model/*.csv .
COPY build_model/backend_code/* backend_code/
COPY docker/flask_a/requirements.txt .
RUN pip install -r requirements.txt
RUN python3 train.py


#Build the production layer
#TODO: I could isolate a virtual python env and copy it from train, to avoid duplication of installing requirments.txt
FROM base
COPY --from=train_model /build_model/model.pkl /build_model/
COPY build_model/backend_code/*.py /build_model/backend_code/
WORKDIR /apps/flask_a
COPY apps/flask_a/ .
COPY docker/flask_a/requirements.txt .
RUN pip install -r requirements.txt

#Run the server by default
#Note: useradd command from https://stackoverflow.com/questions/27701930/how-to-add-users-to-docker-container
RUN useradd --create-home --shell /bin/bash nonroot
USER nonroot
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]


