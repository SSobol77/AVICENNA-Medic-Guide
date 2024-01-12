# We use Python 3.11.7 software
FROM python:3.11.7-slim-buster

# Installing the working directory in the container
WORKDIR /usr/src/app

# Copy the file requirements.txt in the container
COPY requirements.txt ./

# Setting dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files to the container
COPY . .

# Installing NLTK packages
RUN python -m nltk.downloader punkt

# Starting model training
CMD ["python", "./training/train.py"]
