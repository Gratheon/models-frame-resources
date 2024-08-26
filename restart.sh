cd /www/models-frame-resources/
rm -rf /www/models-frame-resources/tmp/*

COMPOSE_PROJECT_NAME=gratheon docker-compose down
COMPOSE_PROJECT_NAME=gratheon docker-compose up --build -d