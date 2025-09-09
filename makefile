.PHONY: start install test clean

# Detect the OS
ifeq ($(OS),Windows_NT)
	ACTIVATE = venv\Scripts\activate &&
else
	ACTIVATE = source venv/bin/activate &&
endif

# Start the development server
start:
	@echo "Starting FastAPI development server..."
	@$(ACTIVATE) uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Install dependencies
install:
	@echo "Installing dependencies..."
	@$(ACTIVATE) pip install -r requirements.txt

# Run tests
test:
	@echo "Running tests..."
	@$(ACTIVATE) python -m pytest

# Clean up
clean:
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete