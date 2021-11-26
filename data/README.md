# Data Engineering Layer
The data engineeing layer is being used to prepare the training data for the later machine learning step.

This part of the processing uses the following libraries
- nummpy and pandas: For general data processing and wrangling
- sqlalchemy: To store data (i.e. disaster messages) in a relational database, which is the data integration hub between the engineering part and machine learning
- nltk: As specific library for natural language processing, i.e. tokenization, removal of stop words, and lemmatization 
