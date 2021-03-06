import json
from abc import ABC
from datetime import date, datetime, time, timedelta

import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from model_template import ModelTemplate


def get_start_and_end_time():
    end = datetime.combine(date.today(), time.min)
    start = end - timedelta(weeks=16)
    return start.timestamp(), end.timestamp()


class ProphetBaselineTemplate(ModelTemplate, ABC):
    def __init__(self):
        """
        ProphetBaseline is an exceptional class. Its train() method also calls predict(), and caches the pred
        This is because in time series prediction, training -> inference is bijective (model is single-use)
        (In most other ML workflows, one training provides a model for multiple inferences)
        Training is asynchronous, so we add the inference there and return it immediately
        """
        super().__init__()
        self.model = None

    def do_train(self, **kwargs):
        # if all data is too sparse (>90% NaN), abort experiment
        useful_data_found = False
        for (node, sensors) in kwargs["items"].items():
            for sensor in sensors:
                data = self.process_data(node, sensor)
                if data is None:
                    print(
                        f"No data at all found for sensor {sensor}, no prediction"
                    )
                    return
                if data["y"].isna().sum() / len(data) > 0.9:
                    print(
                        f"More than 90-% missing data for sensor {sensor} at node {node}, no prediction"
                    )
                    continue
                useful_data_found = True  # great! experiment not aborted
                self.model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                )
                if kwargs["country"] is not None:
                    self.model.add_country_holidays(
                        country_name=kwargs["country"])
                self.model.fit(data)
                print("finished fitting model")

                # metadata = performance_metrics(
                #     cross_validation(self.model,
                #                      initial='7 days',
                #                      period='7 days',
                #                      horizon='12 hours'))
                # for metric in ['mape', 'rmse', 'mae']:
                #     if metric in metadata:
                #         print(f'returning {metric}')
                #         accuracy = metadata[metric].mean()
                #         break

                pred_params = {
                    "baseline_hours": kwargs.get("baseline_hours", 24 * 7),
                    "baseline_only": kwargs.get("baseline_only", True),
                    "from_cache": False,
                }
                pred = self.do_predict({**kwargs, **pred_params})
                if "callback" not in kwargs:
                    kwargs["callback"] = self.client.generate_callback()
                response = json.dumps(
                    {
                        "node": node,
                        "sensor": sensor,
                        # 'accuracy': accuracy,
                        "response": pred,
                    },
                    default=str)
                self.client.notify_model_training_status(
                    response, kwargs["callback"])
        if not useful_data_found:
            self.experiment_aborted = True

    def get_metadata(self):
        df_cv = cross_validation(self.model,
                                 initial="7 days",
                                 period="7 days",
                                 horizon="12 hours")
        metadata = performance_metrics(df_cv)
        return metadata.to_dict()

    def do_predict(self, context):
        if context.get("from_cache", True):
            return self.pred
        future = self.model.make_future_dataframe(4 *
                                                  context["baseline_hours"],
                                                  "15min",
                                                  include_history=False)
        forecast = self.model.predict(future)
        forecast = forecast.set_index("ds")
        forecast.index = (forecast.index.astype(int) // 10**6).astype(
            str)  # for date serialisation
        print("finished prediction")
        # if context['baseline_only']:
        #     return forecast[['yhat']].to_dict()

        # const_2, const_3 = 1.5, 2
        forecast = forecast.rename(columns={
            "yhat_upper": "H1",
            "yhat_lower": "L1"
        }).copy()
        # forecast['H2'], forecast['H3'] = const_2 * \
        # forecast['H1'], const_3 * forecast['H1']
        # forecast['L2'], forecast['L3'] = const_2 * \
        # forecast['L1'], const_3 * forecast['L1']
        # return forecast[['yhat', 'H1', 'H2', 'H3', 'L1', 'L2', 'L3']].to_dict()
        return forecast[["yhat", "H1", "L1"]].to_dict()

    def process_data(self, point_id, prediction_param):
        start, end = get_start_and_end_time()
        data = self.fetch_data(point_id, start, end)
        if data is None:
            return data
        df = pd.DataFrame(data["samplings"]).T[[prediction_param]]
        df.index = pd.to_datetime(df.index, unit="s")
        df = df.sort_index().astype(float)
        return (df.reset_index().rename(columns={
            "index": "ds",
            prediction_param: "y"
        }).copy())
