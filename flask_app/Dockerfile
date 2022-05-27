# The Flask app itself will use some TensorFlow utilities
# and is not responsible for inference. Therefore we do not
# need GPU support.
FROM tensorflow/tensorflow

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]