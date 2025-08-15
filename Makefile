.PHONY: install env augment train test graphs

install:
	pip install -r requirements-min.txt

env:
	set -a; [ -f .env ] && . ./.env || . ./.env.example; set +a

augment:
	set -a; [ -f .env ] && . ./.env || . ./.env.example; set +a; \
	python chat_api_query.py | cat

train:
	set -a; [ -f .env ] && . ./.env || . ./.env.example; set +a; \
	python main.py | cat

test:
	set -a; [ -f .env ] && . ./.env || . ./.env.example; set +a; \
	python - <<'PY'
from main import Trainer
trainer = Trainer()
print(trainer.test())
PY

graphs:
	set -a; [ -f .env ] && . ./.env || . ./.env.example; set +a; \
	python construct_graph.py | cat