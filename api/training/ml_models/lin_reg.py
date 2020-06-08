from abc import ABC

import numpy as np
from sklearn.linear_model import LinearRegression

from model_template import ModelTemplate


class LinRegTemplate(ModelTemplate, ABC):
    def __init__(self):
        self.client = None  # instead of super().__init__() - since we don't need a client
        self.model = LinearRegression()

    def do_train(self, client, context):
        x = np.array(context["X"])
        y = np.dot(x, np.array(context["Y"])) + 3
        self.model.fit(x, y)
        print('finished fitting model')

    def do_predict(self, context):
        return self.model.predict(np.array(context["X_test"])).tolist()
