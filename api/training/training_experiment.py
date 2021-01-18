import json
import os
import pickle
import re
import sys
from importlib import import_module


def save_model(model, path):
    with open(path, "wb+") as f:
        pickle.dump(model, f)
    print(f"Model saved locally at {path}")


def train(params):
    model = params.pop("name")

    camel_to_snake = lambda x: re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower()
    model = camel_to_snake(model)  # handle camel case
    model_module = "ml_models." + model + "_model"
    model_class = model.title().replace("_", "")
    model_module = import_module(model_module)
    my_model = getattr(model_module, model_class)
    m = my_model()
    m.train(**params)
    if m.experiment_aborted:
        return

    # save model, upload model + metadata
    model_name = model  # TODO save old models?
    export_dir = os.path.abspath(
        os.environ.get("PS_MODEL_PATH", os.getcwd() + "../../../models")
    )
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)
    model_path = export_dir + "/" + model_name + ".pkl"
    save_model(m, model_path)
    # m.save_metadata()


if __name__ == "__main__":
    train(json.loads(sys.argv[1]))
