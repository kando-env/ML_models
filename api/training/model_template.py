import json
import os
import time

from dotenv import load_dotenv
from kando import kando_client

load_dotenv()


class ModelTemplate:
    def __init__(self):
        base_url = "https://kando.herokuapp.com"
        self.client = kando_client.client(base_url, os.getenv('KEY'), os.getenv('SECRET'))

    def train(self, **kwargs):
        print('training...')
        start_time = time.time()
        self.do_train(**kwargs)
        print(f'training took {time.time() - start_time} seconds')

    def predict(self, context):
        print('predicting...')
        start_time = time.time()
        pred = self.do_predict(context)
        print(f'predicting took {time.time() - start_time} seconds')
        return pred

    def save_metadata(self):
        print('saving metadata...')
        start_time = time.time()
        metadata = self.get_metadata()
        export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '../../../models'))
        with open(export_dir + '/gradient-model-metadata.json', 'w') as f:
            json.dump(metadata, f)
        print(f'finished, took {time.time() - start_time} seconds')

    def fetch_data(self, point_id, start, end):
        data = self.client.get_all(point_id=point_id, start=start, end=end)
        if len(data['samplings']) == 0:
            print(f'No data found at point {point_id}')
            return None
        return data

    def do_train(self, context):
        raise NotImplementedError

    def do_predict(self, context):
        raise NotImplementedError

    def do_save_metadata(self):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError
