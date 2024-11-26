PARAM_BOUNDS = {
    # RandomForestClassifier and RandomForestRegressor
    # Numerical Parameters
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
    "max_samples": {
        "type": "mixed",  # Supports both int and float
        "int_bounds": [1, 10000],  # Absolute number of samples
        "float_bounds": [0.01, 1.0],  # Fraction of samples
    },
    "ccp_alpha": {"type": "float", "bounds": [0.0, 1.0]},  # Complexity parameter for pruning

    # Categorical Parameters
    "criterion": {"type": "categorical", "options": ["gini", "entropy", "log_loss"]},  # Split quality function
    "max_features": {
        "type": "mixed",  # Supports both categorical and numerical
        "categorical_options": ["sqrt", "log2", None],
        "int_bounds": [1, 100],  # Number of features
        "float_bounds": [0.01, 1.0],  # Fraction of features
    },
    "class_weight": {
        "type": "categorical",
        "options": ["balanced", "balanced_subsample", None],
    },  # Class weights

    # Boolean Parameters
    "bootstrap": {"type": "categorical", "options": [True, False]},  # Bootstrap sampling
    "oob_score": {"type": "categorical", "options": [True, False]},  # Out-of-Bag scoring
    "warm_start": {"type": "categorical", "options": [True, False]},  # Add trees incrementally

    # SVC and NuSVC
    "kernel": {"type": "categorical", "options": ["linear", "poly", "rbf", "sigmoid", "precomputed"]},  # Kernel type
    "degree": {"type": "int", "bounds": [2, 6]},  # Degree of the polynomial kernel
    "C": {"type": "float", "bounds": [1e-3, 1e3]},  # Regularization parameter
    "gamma": {"type": "categorical", "options": ["scale", "auto"]},  # Kernel coefficient
    "probability": {"type": "categorical", "options": [True, False]},  # Whether to enable probability estimates
    "shrinking": {"type": "categorical", "options": [True, False]},  # Whether to use shrinking heuristic
    "decision_function_shape": {"type": "categorical", "options": ["ovo", "ovr"]},  # Multiclass decision shape
    "break_ties": {"type": "categorical", "options": [True, False]},  # Break ties in prediction (if `probability=True`)

    # SVR and NuSVR
    "epsilon": {"type": "float", "bounds": [1e-3, 1.0]},  # Epsilon-tube for support vectors
    "nu": {"type": "float", "bounds": [1e-3, 1.0]},  # Nu parameter for NuSVC and NuSVR

    # Common Parameters for SVM (all types)
    "max_iter": {"type": "int", "bounds": [-1, 10000]},  # Maximum iterations (-1 for no limit)
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
