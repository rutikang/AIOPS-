# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# RUN pip install -r requirements.txt

RUN pip install prometheus_client
RUN pip install requests
RUN pip install pandas
RUN pip install prophet
RUN pip install scikit-learn




# Define environment variable
# ENV NAME monitoring_app

# Run app.py when the container launches
CMD ["python", "monitor.py"]
