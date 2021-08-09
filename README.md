# TOUR

Code for [[WWW'20] _TOUR: Dynamic Topic and Sentiment Analysis of User Reviews for Assisting App Release_](https://dl.acm.org/doi/10.1145/3442442.3458612).

Visit [the project homepage](https://yttty.github.io/tour/) for video demonstration and user feedbacks.

### Installation

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm` or `conda install -c conda-forge spacy-model-en_core_web_sm`
3. `pip install Cython` or `conda install Cython`
4. `python build_pyx.py build_ext --inplace`
5. `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt')"`


### Usage

1. `mkdir results`
2. `python app.py`
3. visit http://localhost:5000
