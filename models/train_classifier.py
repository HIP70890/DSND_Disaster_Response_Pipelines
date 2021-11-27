import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath: str):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)

    # fetch the message feature as input
    X = df['message']
    # use all other features as labels, while dropping 'orginal' 
    # (ie. message in native language), and 'genre'
    Y = df.drop(['message', 'id', 'original', 'genre', 'child_alone'], axis=1)

    pd.set_option('display.max_columns', None)
        
    return X, Y, Y.columns

# initialize single lemmatizer
lemmatizer = WordNetLemmatizer()

# create a tokenizer funtion which will tokenize sentenses, dropping punctuation, and removing stop words
def tokenize(text: str) -> list:
    # normalize, tokenize, and only select alphanumeric tokens
    words = [word for word in word_tokenize(text.lower()) if word.isalnum()]
    # remove stopwords (english) and lemmatize
    return [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

# Build the model, based on the results of the exploration
def build_model():
    rfc = RandomForestClassifier(random_state=42)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(rfc, n_jobs=-1))
    ])

    parameters = {
        'vect__max_df': (0.8, 0.9),
        'mclf__estimator__min_samples_split': [6, 8],
        'mclf__estimator__n_estimators': [10, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    return cv


def evaluate_model(model, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: list):
    y_pred = model.predict(X_test)

    # print score per feature
    for i, column in enumerate(category_names):
        print(column)
        print(classification_report(Y_test[column], y_pred[:, i]))

    # print accuracy
    accuracy = (y_pred == Y_test.values).mean()
    print('Model accuracy is {:.3f}'.format(accuracy))

        
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()