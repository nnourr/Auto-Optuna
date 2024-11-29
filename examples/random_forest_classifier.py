from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import is_classifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, check_scoring, r2_score

def get_data():
    """Fetches preprocessed data."""
    return datasets.load_breast_cancer(return_X_y=True)

def get_model():
    """Returns an uninitialized model."""
    return RandomForestClassifier

def train_baseline():
  X, y = get_data()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) # set random state for control
  
  classifier = get_model()
  model = classifier()
  
  model.fit(X_train, y_train)
  
  scorer = accuracy_score if is_classifier(classifier) else r2_score
  
  predictions = model.predict(X_test)
  accuracy = scorer(y_test, predictions)
  print(accuracy)

if __name__ == "__main__":
  train_baseline()