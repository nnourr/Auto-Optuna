from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_data():
    """Fetches preprocessed data."""
    return datasets.load_breast_cancer(return_X_y=True)

def get_model():
    """Returns an uninitialized model."""
    return DecisionTreeClassifier  # Notice, we return the class, not an instance.

def train_baseline():
  X, y = get_data()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) # set random state for control
  
  classifier = get_model()
  model = classifier()
  
  model.fit(X_train, y_train)
  
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  print(accuracy)

if __name__ == "__main__":
  train_baseline()