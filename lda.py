from logger import Logger
from typing import List

# Spacy
import spacy

# Gensim
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# plot coherence or not
plot_coherence = False
vis_topics = False

logger = Logger("Coherence LDA")

# spacy
nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS
my_stopwords = [
    "let's", "i'd", "he'd", 'would', "i'll", "why's", "they've", 'could',
    "how's", "we're", "we'll", "that's", "he'll", "he's", "who's", "she'd",
    "there's", "i've", "they'd", "here's", "we've", "where's", "what's",
    'ought', "can't", "they'll", 'cannot', "i'm", "they're", "we'd", "when's",
    "she'll"
]
stop_words.union(my_stopwords)


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[
        word for word in simple_preprocess(str(doc)) if word not in stop_words
    ] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def compute_coherence_values(dictionary, corpus, texts, num_topics):
    """
    Compute c_v coherence for topic model

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    num_topics : number of topics

    Returns:
    -------
    model_list : LDA topic model
    coherence_values : Coherence values corresponding to the LDA model
    """
    # Build LDA model
    model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
    coherencemodel = CoherenceModel(model=model,
                                    texts=texts,
                                    dictionary=dictionary,
                                    coherence='c_v')

    return model, coherencemodel.get_coherence()


def coherence_lda(review: List[str],
                  num_topics: int = 4,
                  num_topic_words: int = 30,
                  gram_mode: str = "bigram"):
    """
    LDA Model which automatically determines the number of topics

    Parameters:
    ----------
    review : List of documents in str format
    num_topics : topic number

    Returns:
    -------
    lda_model : best LDA model
    coherence_values : Coherence values corresponding to the LDA model
    doc_topics : probability of topics for each document, List[List[float]]
    topic_words : words in each topic with significance
    """
    assert gram_mode in ['bigram', 'trigram']
    data_words = list(sent_to_words(review))
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        data_words, min_count=5,
        threshold=100)  # higher threshold fewer phrases.
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    if gram_mode == 'trigram':
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        data_words_bigrams = make_trigrams(data_words_nostops)
    else:
        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(
        data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words_bigrams)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    logger.info(f'Number of documents: {len(corpus)}')

    # select n_topics by coherence
    lda_model, coherence_value = compute_coherence_values(
        dictionary=id2word,
        corpus=corpus,
        texts=data_lemmatized,
        num_topics=num_topics)

    topic_words = lda_model.show_topics(num_topics=-1,
                                        num_words=num_topic_words,
                                        formatted=False)
    # doc_lda = lda_model[corpus]

    # Compute Perplexity: a measure of how good the model is. lower the better.
    logger.info(f'Perplexity: {lda_model.log_perplexity(corpus)}')
    logger.info(f'Coherence Score: {coherence_value}')

    doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]

    return lda_model, coherence_value, doc_topics, topic_words
