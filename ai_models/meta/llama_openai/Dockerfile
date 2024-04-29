# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Flask application code into the container
COPY . /app

# Install Flask and other dependencies
RUN pip install --no-cache-dir Flask

# Expose the port on which the Flask app will run (e.g., 5000)
EXPOSE 5000

# Set environment variables (optional)
ENV FLASK_APP=routes.py
ENV FLASK_RUN_HOST=0.0.0.0

# Define the command to run the Flask application
CMD ["flask", "run"]
