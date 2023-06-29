from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import nltk
from nltk.corpus import stopwords
import joblib
import pymorphy2


# Read the data from joblib
def read_data(file1, file2):
    urgent = joblib.load(file1)
    non_urgent = joblib.load(file2)

    urgent_df = pd.DataFrame(urgent, columns=['case'])
    non_urgent_df = pd.DataFrame(non_urgent, columns=['case'])

    urgent_df = urgent_df.assign(is_urgent=[1 for i in range(len(urgent))])
    non_urgent_df = non_urgent_df.assign(is_urgent=[0 for i in range(len(non_urgent))])

    df = pd.concat([urgent_df, non_urgent_df], axis=0)
    
    return df


# Data normalization
def normalize(case):
    case = case.str.lower()
    case = case.str.replace('[^\w\s]',' ')
   
    hello_words = ['здравствуйте', 'добрый день', 'привет', 'доктор']
    for w in hello_words:
        case = case.apply(lambda x: x.replace(w, ''))
    
    return case


# Lemmatization
def lemmatize(case):
    morph = pymorphy2.MorphAnalyzer()

    words = case.split() # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return res


# Stop words deletion
def delete_stopwords(case):
    nltk.download('stopwords')
    sw = stopwords.words('russian')

    sw.remove('не')
    sw.remove('нет')

    case = case.apply(lambda x: [word for word in x if not word in sw])

    return case


# Split the dataframe into test and train data
def split_data(df):
    X = df.case
    y = df.is_urgent

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    
    return data


# Train the model, return the model
def train_model(data):
    model = Pipeline([
                ('clf', LogisticRegression()),
                ])
    model.fit(data["train"]["X"], data["train"]["y"])

    return model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    y_predicted = model.predict_proba(data["test"]["X"])[:, 1] > 0.6

    accuracy = accuracy_score(data["test"]["y"], y_predicted)
    f1 = f1_score(y_predicted, data["test"]["y"])
    precision = precision_score(y_predicted, data["test"]["y"])
    recall = recall_score(y_predicted, data["test"]["y"])

    metrics = {"accuracy": accuracy,
               "f1": f1,
               "precision": precision,
               "recall": recall}
    
    return metrics


def main():
    # Load Data
    df = read_data("urgent.joblib", "non-urgent.joblib")
    
    # Preprocessing
    df.case = normalize(df.case)
    df.case = df.case.apply(lambda x: lemmatize(x))
    df.case = delete_stopwords(df.case)
    
    # Split Data into Training and Test Sets
    data = split_data(df)

    # Train Model on Training Set
    model = train_model(data)

    # Validate Model on Test Set
    metrics = get_model_metrics(model, data)

    # Save Model
    model_name = "urgency_classifier.pkl"
    joblib.dump(value=model, filename=model_name)


if __name__ == '__main__':
    main()