install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 main.py --ignore=E226,E501

test:
	python -m pytest -vv test_main.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test

