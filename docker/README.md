This folder contains helper Docker artifacts to run the repo services for development.

What is included:
- `Dockerfile-notebook`: image for running Jupyter Lab with repository mounted for interactive work.
- `Dockerfile-api`: image for running an API (uvicorn) exposed on port 8000.
- `docker-compose.yml`: runs both services together for local dev.

Quick start (PowerShell)

# build images and start services
docker compose -f docker/docker-compose.yml up --build

# stop services
docker compose -f docker/docker-compose.yml down

Notes:
- The compose file mounts the repository into the containers so changes are reflected immediately (useful for development). For production, build immutable images and avoid mounting source code.
- If your API module path differs update the `CMD` in `docker/Dockerfile-api` to point to the correct module, e.g. `projects/02-cardiovascular-risk-ml.src.api.main:app`.
- To enable interactive Plotly in notebooks, install required browser plugins or open the saved HTML files created by the notebook cells.

Security and data:
- This setup is for local development only. Do not publish containers that contain private data or secrets. Add secrets via environment variables or secret managers for production.

If you want, I can:
- Create a lightweight production Dockerfile for the API (smaller base image, non-root user).
- Adjust the compose services to run selected project APIs by name.
- Build and run these images here (I can't execute Docker locally in this environment) and verify logs—tell me which service to focus on.
