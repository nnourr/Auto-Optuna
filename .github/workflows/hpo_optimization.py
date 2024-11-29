import importlib.util
import optuna
from pathlib import Path
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, r2_score
import os

from sklearn.model_selection import train_test_split

from params import PARAM_BOUNDS, validate_hyperparameters, ignore_param

def load_script(script_path, get_data_method, get_model_method):
    """Dynamically load a Python module and fetch specified methods."""
    spec = importlib.util.spec_from_file_location("script_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Fetch the methods dynamically
    get_data = getattr(module, get_data_method)
    get_model = getattr(module, get_model_method)

    return {"get_data": get_data, "get_model": get_model}

def suggest_hyperparameter(trial, param_name, config, model_name):
  """Suggest a hyperparameter based on its configuration."""
  if config["type"] == "int":
      return trial.suggest_int(param_name, *config["bounds"])
  elif config["type"] == "float":
      return trial.suggest_float(param_name, *config["bounds"])
  elif config["type"] == "categorical":
      if isinstance(config["options"], dict) and model_name:
          options = config["options"].get(model_name, [])
          return trial.suggest_categorical(param_name, options)
      return trial.suggest_categorical(param_name, config["options"])
  elif config["type"] == "mixed":
      if "int_bounds" in config:
          return trial.suggest_int(param_name, *config["int_bounds"])
      elif "float_bounds" in config:
          return trial.suggest_float(param_name, *config["float_bounds"])

def objective(trial, script, X_train, X_test, y_train, y_test):
    """Objective function for hyperparameter optimization."""
    # Fetch data and uninitialized model dynamically from the script
    model_class = script["get_model"]()
    model = model_class()  # Instantiate the model
    scorer = accuracy_score if is_classifier(model_class) else r2_score

    # Suggest hyperparameters dynamically based on PARAM_BOUNDS
    params = model.get_params()
    model_name = model_class.__name__
    suggested_params = {}
    for param_name, param_value in params.items():
      if param_name in PARAM_BOUNDS:
          suggested_params[param_name] = suggest_hyperparameter(trial, param_name, PARAM_BOUNDS[param_name], model_name)
      elif not ignore_param(param_name):
        print("WARNING: Unrecognized hyperparameter: " + param_name)
        suggested_params[param_name] = param_value

    # Validate parameters
    if not validate_hyperparameters(suggested_params):
        return float("-inf")  # Penal
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = scorer(y_test, predictions)
    
    return accuracy


def run_hpo(script_path, get_data_method, get_model_method, n_trials, artifact_name):
    """Run hyperparameter optimization with a dynamically loaded script."""
    script = load_script(script_path, get_data_method, get_model_method)

    # Fetch data
    X, y = script["get_data"]()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    model_class = script["get_model"]()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, script, X_train, X_test, y_train, y_test), n_trials=n_trials, n_jobs=-1)

    # Train the final model with best hyperparameters
    best_params = study.best_params
    validate_hyperparameters(best_params)
    final_model = model_class(**best_params)
    final_model.fit(X, y)

    # Save the model
    with open(artifact_name, "wb") as f:
        import pickle
        pickle.dump(final_model, f)

    print("Best Parameters:", best_params)
    print("Best Accuracy:", study.best_value)

if __name__ == "__main__":
    script_path = os.getenv("INPUT_SCRIPT_PATH")
    get_data_method = os.getenv("INPUT_GET_DATA_METHOD", "get_data")
    get_model_method = os.getenv("INPUT_GET_MODEL_METHOD", "get_model")
    n_trials = int(os.getenv("INPUT_N_TRIALS", "20"))
    artifact_name = os.getenv("INPUT_ARTIFACT_NAME", "optimized_model.pkl")

    run_hpo(script_path, get_data_method, get_model_method, n_trials, artifact_name)