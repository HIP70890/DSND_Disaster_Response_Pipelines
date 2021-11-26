# Disaster Response Pipeline Project
This project was created as a submission for the Data Science Nanodegree program of Udacity.
It implements a simple data engineering and machine learning pipeline for categorizing text messages collect in the context of natural disasters into different subject areas. This should allow dispatching the messages to difference disaster response agencies, in order to optimize the processing of notifications and disaster interventions.

## General Idea
The general idea of the solution is that messages created by impacted people on the ground (i.e. in the disaster area) will use common words to describe intentions and contexts, which should allow a natural language processing pipeline to be trained on those types of messages, and allow it to assign each message to one or more categories, which in turn are being observed by different recipients for preparation of actions.

The solution was being provided with about 26k messages, labeled into one or more of 35 different categories. These messages will be analyzed with NLP processing libraries, to extract keywords (and e.g. all stop words removed). These words are then encoded into ML ready features, and together with the labels, are being fed into a ML pipeline for training and testing. 
## Software Architecture
The solution is a completelly python-based implementation, that uses different open source software packages. It has a simple three layered approach for processing:
1. Data engineering
2. Machine learning
3. Web presentation, and human interface
### Data Engineering
There is a Jupyter notebook that was being used to explore the data and prepare it for ML: [ETL Pipeline Preparation.ipynb](notebooks/ETL%20Pipeline%20Preparation.ipynb).

Furthermore, the pipeline has been transformed into a python file: [process_data.py](data/process_data.py)
### Machine Learning
Again, ML has been explored via a dedicated Jupyter notebook: [ML Pipeline Preparation.ipynb](notebooks/ML%20Pipeline%20Preparation.ipynb)

The final ML implementation is in the models directory: [train_classifier.py](models/train_classifier.py)

### Web presentation and human interface
- Flask: For implementing the web presentation of ML predictions, and as a processor of new text messages from the human interface 
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
