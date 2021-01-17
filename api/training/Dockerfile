FROM python:3.7
COPY requirements.txt /tmp/

ARG GITHUB_ACCESS_TOKEN

RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt
RUN pip install fbprophet
RUN pip install xgboost
RUN pip install git+https://$GITHUB_ACCESS_TOKEN@github.com/nadavk72/kando-python-client.git
CMD ["sh"]
