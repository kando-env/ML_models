# syntax = docker/dockerfile:1.0-experimental
FROM python:3.7
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt
RUN pip install fbprophet
RUN pip install xgboost
# multi step build - add local shh key and securely pip install private repo. the private key isn't stored in the build
# https://stackoverflow.com/a/62258239/6220759
# RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
# RUN --mount=type=ssh pip install git+ssh://git@github.com/nadavk72/kando-python-client.git
RUN pip install git+https://f0e06b4b4e20f0d9209cd7f4b3e0473d26ecba59@github.com/nadavk72/kando-python-client.git@staging
CMD ["sh"]
CMD ["echo "]
