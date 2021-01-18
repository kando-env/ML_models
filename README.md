# ML Models

This repo contains the model code for Kando/Gradient.
Add new models to here to run as Gradient experiments via the ML training server, or `model_runner` library.
If you want to run a model locally, use `local_experiment.py`

**Rules for well-behaved models:**
1) A model _must inherit_ from ModelTemplate
2) A model _must implement_ a `do_train` and `do_predict` method
3) A model name _must be identical_ to its module name, modulo case conventions and the suffix `_model`!
For example: if the model name is `CodLstmSmall`, the file name _must be_ `cod_lstm_small_model.py`.

Have fun! 
