FROM kandoenv/training

COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt
COPY inference_server.py /code/
COPY training /code/training
CMD ["python", "/code/inference_server.py"]
