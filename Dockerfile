# Use the official Python slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the files from the repository into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add the entrypoint for the action
ENTRYPOINT ["python", "/app/.hpo_optimization.py"]
