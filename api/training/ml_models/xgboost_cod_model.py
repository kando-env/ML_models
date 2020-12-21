import numpy as np
import pandas as pd
import xgboost as xgb
from model_template import ModelTemplate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split


class XgboostCod(ModelTemplate):
    def __init__(self):
        super().__init__()
        self.xgbr = xgb.XGBRegressor()
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = (
            None,
            None,
            None,
            None,
            None,
        )

    def do_train(self, **kwargs):
        x, y = self._process_data(**kwargs)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.15)
        _ = self.xgbr.fit(self.X_train, self.y_train)
        print("finished fitting model")

    def get_metadata(self):
        self.y_pred = self.xgbr.predict(self.X_test)
        kfold = KFold(n_splits=10, shuffle=True)
        metadata = {
            "scores":
            cross_val_score(self.xgbr, self.X_train, self.y_train,
                            cv=5).tolist(),
            "kf_cv_scores":
            cross_val_score(self.xgbr, self.X_train, self.y_train,
                            cv=kfold).tolist(),
            "mse":
            mean_squared_error(self.y_test, self.y_pred),
            "rmse":
            np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
        }
        return metadata

    def do_predict(self, context):
        # list to convert to json
        return self.xgbr.predict(self.X_test).tolist()

    def _process_data(self, **kwargs):
        data = self.fetch_data(kwargs["point_id"], kwargs["start"],
                               kwargs["end"])
        df = (pd.DataFrame(data['samplings']).T[[
            "EC", "PH", "COD", "TSS", "FLOW", "TEMPERATURE"
        ]].fillna(method="ffill"))
        x, y = df[["EC", "PH", "TSS", "FLOW",
                   "TEMPERATURE"]].copy(), df.COD.copy()
        return x, y
