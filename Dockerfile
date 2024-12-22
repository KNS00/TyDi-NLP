# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies, using pip cache
COPY requirements.txt /app/ 
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project into the container
COPY . /app/

# Expose the Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter-notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--notebook-dir=/app/notebooks"]
