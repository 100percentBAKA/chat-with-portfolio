# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Copy the .env file into the container
COPY .env .env

# Start the FastAPI application using a script to source the .env file
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

