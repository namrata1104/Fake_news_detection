gar_creation:

"""
Configures Docker to work with Google Cloud (gcloud auth configure-docker)
Creates a Google Artifact Registry repository (gcloud artifacts repositories create)
Uses environment variables like GCP_REGION, GAR_REPO, and GCP_PROJECT for configurations
"""

	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
	gcloud artifacts repositories create ${GAR_REPO} --repository-format=docker \
	--location=${GCP_REGION} --description="Repository for storing ${GAR_REPO} images"

docker_build:

"""Builds a Docker image for the project"""

	docker build --platform linux/amd64 -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod .

docker_push:

"""Pushes the built Docker image to the Google Artifact Registry"""

	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod

docker_run:

"""Runs the Docker image locally as a container"""

	docker run -e PORT=8000 -p 8000:8000 --env-file .env ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod

docker_interactive:

"""Runs the Docker image interactively, useful for debugging"""

	docker run -it --env-file .env ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod /bin/bash

docker_deploy:

"""Deploys the Docker image to Google Cloud Run"""

	gcloud run deploy --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod --memory ${GAR_MEMORY} --region ${GCP_REGION}


run_api:

"""Runs the API locally using Uvicorn"""

	uvicorn fake_news_detection.api.fast:app --reload

to start the backend excecute:
uvicorn fake_news_detection.api.fast:app --reload

to start the frontend change to the frontend folder and execute this command:
streamlit run app.py
