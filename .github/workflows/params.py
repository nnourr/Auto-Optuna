PARAM_BOUNDS = {
    # RandomForestClassifier and RandomForestRegressor
    "n_estimators": {"type": "int", "bounds": [10, 300]},  # Number of trees in the forest
    "max_depth": {"type": "int", "bounds": [1, 100]},  # Maximum depth of the tree
    "min_samples_split": {
        "type": "mixed",  # Supports both int and float
        "int_bounds": [2, 20],
        "float_bounds": [0.01, 1.0],  # Fraction of samples (0.01–1.0)
    },
    "min_samples_leaf": {
        "type": "mixed",  # Supports both int and float
        "int_bounds": [1, 20],
        "float_bounds": [0.01, 0.5],  # Fraction of samples (0.01–0.5)
    },
    "min_weight_fraction_leaf": {"type": "float", "bounds": [0.0, 0.5]},  # Fraction of total weights
    "max_leaf_nodes": {"type": "int", "bounds": [2, 1000]},  # Maximum number of leaf nodes
    "min_impurity_decrease": {"type": "float", "bounds": [0.0, 1.0]},  # Impurity threshold
    "ccp_alpha": {"type": "float", "bounds": [0.0, 1.0]},  # Complexity parameter for pruning
    "bootstrap": {"type": "categorical", "options": [True, False]},  # Bootstrap sampling
    "oob_score": {"type": "categorical", "options": [True, False]},  # Out-of-Bag scoring
    "warm_start": {"type": "categorical", "options": [True, False]},  # Add trees incrementally

    # DecisionTreeClassifier and DecisionTreeRegressor
    "criterion": {
        "type": "categorical",
        "options": {
            "DecisionTreeClassifier": ["gini", "entropy", "log_loss"],
            "DecisionTreeRegressor": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "RandomForestClassifier": ["gini", "entropy", "log_loss"]
        },
    },  # Split quality function
    "class_weight": {
        "type": "categorical",
        "options": {
            "DecisionTreeClassifier": ["balanced", None],  # Only applicable to classifiers
            "RandomForestClassifier":["balanced", "balanced_subsample", None]
        },
    },  # Class weights
    "splitter": {"type": "categorical", "options": ["best", "random"]},  # Split strategy
    "max_features": {
        "type": "mixed",  # Supports both categorical and numerical
        "categorical_options": ["sqrt", "log2", None],
        "int_bounds": [1, 100],  # Number of features
        "float_bounds": [0.01, 1.0],  # Fraction of features
    },
    "max_depth": {"type": "int", "bounds": [1, 100]},  # Maximum depth of the tree
    "min_samples_split": {
        "type": "mixed",  # Supports both int and float
        "int_bounds": [2, 20],
        "float_bounds": [0.01, 1.0],  # Fraction of samples (0.01–1.0)
    },
    "min_samples_leaf": {
        "type": "mixed",  # Supports both int and float
        "int_bounds": [1, 20],
        "float_bounds": [0.01, 0.5],  # Fraction of samples (0.01–0.5)
    },
    "min_impurity_decrease": {"type": "float", "bounds": [0.0, 1.0]},  # Impurity threshold
    "ccp_alpha": {"type": "float", "bounds": [0.0, 1.0]},  # Complexity parameter for pruning
    "max_leaf_nodes": {"type": "int", "bounds": [2, 1000]},  # Maximum number of leaf nodes
}


def validate_hyperparameters(params):
    if not params.get("bootstrap", True):
        if params.get("max_samples", False): params["max_samples"] = None
        if params.get("oob_score", False): params["oob_score"] = False
    if params.get("max_samples", False):
        if not params.get("bootstrap", True): params["max_samples"] = None
    return True

def ignore_param(param):
    ignored = ['verbose', 'n_jobs', 'monotonic_cst', 'random_state']
    return param in ignored
