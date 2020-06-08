FROM python:3.7
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt
RUN pip install fbprophet
RUN pip install xgboost
RUN pip install git+https://5f9e11f927df8d9f896be51cbc2f1c2d17a2c90d@github.com/nadavk72/kando-python-client.git
CMD ["sh"]
