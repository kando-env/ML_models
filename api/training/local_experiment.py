from datetime import datetime

from api.training.training_experiment import train

params = {
    'name': 'xgboost_cod',
    'point_id': 1012,
    "start": datetime(2020, 11, 10, 3, 55).timestamp(),
    "end": datetime(2020, 12, 10, 3, 55).timestamp()
}
model = train(params)
