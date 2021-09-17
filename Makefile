docker_run: docker_build
	docker run --env DISCORD_TOKEN="$${DISCORD_TOKEN}" --env DISCORD_EXEC=1 --mount type=bind,source=$$(pwd)/ctxs,destination=/discord_bot/ctxs bulkyxbot
docker_build:
	docker build -t bulkyxbot .
