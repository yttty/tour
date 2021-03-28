# TOUR

Code for WWW'20 paper "_TOUR: Dynamic Topic and Sentiment Analysis of User Reviews for Assisting App Release_"


### Installation

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm` or `conda install -c conda-forge spacy-model-en_core_web_sm`
3. `pip install Cython` or `conda install Cython`
4. `python build_pyx.py build_ext --inplace`
5. `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt')"`

