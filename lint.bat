@echo off
echo Running isort to organize imports...
python -m isort .

echo Running black code formatter...
python -m black .

echo Running ruff for linting and fixing...
python -m ruff check --fix .

echo Running mypy for type checking...
python -m mypy --exclude build trajectolab

echo Linting and formatting complete!
