docker_run: docker_build
	docker run --env DISCORD_TOKEN="$${DISCORD_TOKEN}" --env DOCKER_EXEC=1 bulkyxbot
docker_build:
	docker build -t bulkyxbot .
