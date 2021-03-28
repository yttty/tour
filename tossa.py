# -*- coding: utf-8 -*-

import os
import re
import pickle
from typing import List, Any
from itertools import groupby
from multiprocessing import Process
import numpy as np
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from gensim.models import word2vec

from lda import coherence_lda
from logger import Logger
from config import COLORS, Review


class Preprocessor:
    """
    Generally preprocess raw reviews
    1. clean sentence
    2. lemmatize
    3. filter irregular words and stopwords
    """
    def __init__(self,
                 user_stop_list: List[str] = None,
                 special_words: List[str] = ['prosses', 'acsses']):
        self.stop_list = stopwords.words('english')
        # special stopwords, build from experience
        additional_stopwords = [
            "let's", "i'd", "he'd", 'would', "i'll", "why's", "they've",
            'could', "how's", "we're", "we'll", "that's", "he'll", "he's",
            "who's", "she'd", "there's", "i've", "they'd", "here's", "we've",
            "where's", "what's", 'ought', "can't", "they'll", 'cannot', "i'm",
            "they're", "we'd", "when's", "she'll"
        ]
        self.stop_list += additional_stopwords
        if user_stop_list:
            self.stop_list += user_stop_list

        # special words to skip lemmatization, build from experience
        self.special_words = special_words

    def clean_sentence(self, sentence: str) -> str:
        """
        Clean review content

        Parameters:
        sentence (str): a sentence in string format

        Returns:
        str: cleaned sentence
        """
        sentence = re.sub('_', '', sentence.lower())
        sentence = re.sub(r'[~;]+', '.', sentence)
        sentence = re.sub(r"[']+", r"'", sentence)
        sentence = re.sub(r"[^0-9a-z,.?!' ]", ' ', sentence)
        sentence = re.sub(r"([,.?!])\1+", r' \1 ', sentence)
        sentence = re.sub(r"([,.?!])", r' \1 ', sentence)
        sentence = re.sub(r'[ ]+', ' ', sentence.strip())
        if re.search(r'\w$', sentence):
            sentence += ' . '
        return sentence

    def lemmatize(self, token: str) -> str:
        """
        Lemmatize review content

        Parameters:
        token (str): a token in string format

        Returns:
        str: lemmatized token
        """
        lemmatizer = WordNetLemmatizer()
        token = re.sub(r"^'|'$", r'', token)
        token = re.sub(r'(.)\1{2,}', r'\1', token)
        token = lemmatizer.lemmatize(token, 'v')
        if token not in self.special_words:
            if (len(token) > 3 and token != lemmatizer.lemmatize(token, 'n')
                    and not re.search(r'ss$', token)):
                token = lemmatizer.lemmatize(token, 'n')
        return token

    def filter_word(self, token: str):
        return (token not in self.stop_list) and token.isalnum() and (
            not len(token) > 15)

    def process(self, review: str):
        return re.sub(
            r'(. )\1+', r'\1',
            re.sub(
                r'[ ]+', ' ', ' '.join(
                    list(
                        filter(self.filter_word, [
                            x[0] for x in groupby(
                                list(
                                    map(
                                        self.lemmatize,
                                        word_tokenize(
                                            self.clean_sentence(review)))))
                        ]))))).strip()


class TopicSentimentAnalysis:
    """
    process the uploaded review and save intermediate result into /intermediate
    param: root_path flask root path
    param: file_path path to review file
    """
    def __init__(self,
                 root_path: str,
                 file_path: str,
                 save_intermediate: bool = True,
                 intermediate_path: str = "intermediate_result",
                 selected_review_fname: str = "selected_review",
                 topic_summary_fname: str = "topic_summary"):
        self.root_path = root_path
        self.file_path = file_path
        self.save_intermediate = save_intermediate
        self.intermediate_path = intermediate_path
        self.selected_review_fname = selected_review_fname
        self.topic_summary_fname = topic_summary_fname
        if not os.path.isdir(self.intermediate_path):
            os.mkdir(self.intermediate_path)
        self.log = Logger(self.__class__.__name__)

    def _dump_intermediate(self, fname: str, obj: Any):
        """
        :param fname: fname filename e.g. "abc.pkl"
        :param obj: obj the object to save
        """
        path = os.path.join(self.root_path, self.intermediate_path, fname)
        with open(path, 'wb') as fout:
            pickle.dump(obj, fout)

    def read_file(self):
        self.reviews = []
        with open(self.file_path, 'r') as f:
            rows = f.readlines()
            line_count = 0
            for r in rows:
                row = r.split("******")
                # filter unknown version
                if row[4] != 'Unknown':
                    self.reviews.append(
                        Review(row[0], row[1], row[2], row[3], row[4], row[5]))
                    line_count += 1
            self.log.info(f"{line_count} reviews in total")

    def get_version(self) -> List[str]:
        ver_list = list(dict.fromkeys(map(lambda x: x.version, self.reviews)))
        return ver_list

    def preprocess(self, version: str = None):
        self.log.info('Preprocess...')
        preprocessor = Preprocessor()
        self.preprocessed_reviews = [
            preprocessor.process(review) for review in map(
                lambda x: x.body,
                filter(lambda x, version=version: version is None or x.version
                       == version,
                       self.reviews))
        ]
        # save preprocessed reviews (for w2v)
        self.preprocessed_file_path = os.path.join(self.root_path,
                                                   self.intermediate_path,
                                                   "preprocessed.txt")
        with open(self.preprocessed_file_path, "w") as fout:
            fout.writelines(
                [review + "\n" for review in self.preprocessed_reviews])

    def parse_dependency(self):
        self.log.info('parse_dependency...')

        def call_stanford_dependency(in_path, out_path):
            os.system(
                f'''java -Xmx1g edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat "typedDependencies" englishPCFG.ser.gz
                {in_path} > {out_path}''')

        self.dependency_file_path = os.path.join(self.root_path,
                                                 self.intermediate_path,
                                                 "dependency.txt")
        self.proc_dependency = Process(target=call_stanford_dependency,
                                       args=(self.preprocessed_file_path,
                                             self.dependency_file_path))
        self.proc_dependency.start()

    def build_w2v_model(self, save_w2v=True):
        self.log.info('Building w2v model')
        # min_count = 5 by default
        sentences = word2vec.Text8Corpus(self.preprocessed_file_path)
        self.model = word2vec.Word2Vec(sentences, size=100)
        if self.save_intermediate and save_w2v:
            self.model_path = os.path.join(self.root_path,
                                           self.intermediate_path, "w2v_model")
            self.model.save(self.model_path)

    def calc_senti_words(self,
                         pwords: List[str] = None,
                         nwords: List[str] = None):
        self.log.info('calc_senti_words')

        # default value of pwords and nwords
        #if not pwords:
        #pwords = ['bad']
        #pwords = ['good']
        #if not nwords:
        #nwords = ['good']
        #nwords = ['bad']

        self.sent = {}
        for wi in self.model.wv.index2word:
            pdist = 0
            ndist = 0
            for word in pwords:
                if word in self.model.wv.vocab:
                    pdist = self.model.wv.similarity(
                        wi, word) if pdist < self.model.wv.similarity(
                            wi, word) else pdist
            for word in nwords:
                if word in self.model.wv.vocab:
                    ndist = self.model.wv.similarity(
                        wi, word) if ndist < self.model.wv.similarity(
                            wi, word) else ndist
            #self.sent[wi] = ndist - pdist
            self.sent[wi] = pdist - ndist
        self.log.info(
            f'{len(self.model.wv.index2word)} words in w2v embedding space')
        if self.save_intermediate:
            self._dump_intermediate("sent.pkl", self.sent)

    def map_dependency(self):
        self.log.info('map_dependency')
        # feature_dep = ('nsubj', 'amod', 'dobj')
        self.proc_dependency.join()

        f = open(self.dependency_file_path, 'r')
        dependency_data = f.readlines()

        for line in dependency_data:
            if line.split('(')[0] == 'nsubj':
                dep = line.split('(')[1][:-1].split(', ')
                self.sent[dep[1].split('-')
                          [0]] = (self.sent[dep[0].split('-')[0]] +
                                  self.sent[dep[1].split('-')[0]]) / 2
            elif line.split('(')[0] == 'amod':
                dep = line.split('(')[1][:-1].split(', ')
                self.sent[dep[0].split('-')
                          [0]] = (self.sent[dep[0].split('-')[0]] +
                                  self.sent[dep[1].split('-')[0]]) / 2
            elif line.split('(')[0] == 'dobj':
                dep = line.split('(')[1][:-1].split(', ')
                self.sent[dep[1].split('-')
                          [0]] = (self.sent[dep[0].split('-')[0]] +
                                  self.sent[dep[1].split('-')[0]]) / 2

    def topic_modeling(self, n_topics: int, gram_mode: str):
        self.log.info('topic_modeling')

        # n_topics: number of topics
        # topic_words: [(topic_idx, [(word, weight),...]),...]
        # doc_topics: [[(topic_idx, probability),...],...]
        self.n_topics = n_topics
        _, self.coherence_value, self.doc_topics, self.topic_words = coherence_lda(
            review=self.preprocessed_reviews,
            num_topics=n_topics,
            gram_mode=gram_mode)
        if self.save_intermediate:
            self._dump_intermediate("coherence_value.pkl",
                                    self.coherence_value)
            self._dump_intermediate("doc_topics.pkl", self.doc_topics)
            self._dump_intermediate("topic_words.pkl", self.topic_words)

    def prepare_review_list(self, probability_threshold: float = 0.3):
        """
        Now prepare data for topic.html

        For each topic_idx, select `number_of_selection` reviews and sort
        reviews by probability. `selected_review` is a list of `n_topics`
        sublists which contains `number_of_selection` review tuples of
        (topic_idx, probability, review text)

        Only select review whose probability of belonging to a topic is greater
        than probability_threshold.
        """

        selected_review = []

        # `all_doc_topics` is a list of tuples consisting of
        # [(topic_idx, probability, review text), ... ]
        # MUST convert to list
        all_doc_topics = list(
            map(
                lambda x: (x[1][0], x[1][1], self.reviews[x[0]]),
                enumerate(
                    map(
                        lambda prob_list: sorted(
                            prob_list, key=lambda x: x[1], reverse=True)[0],
                        self.doc_topics))))
        for topic_idx in range(len(self.topic_words)):
            topic_x = sorted(filter(
                lambda x: True if x[0] == topic_idx and x[1] >
                probability_threshold else False, all_doc_topics),
                             key=lambda x: x[1],
                             reverse=True)
            selected_review.append(topic_x)

        # This is compulsory because we need to show reviews in summary
        self._dump_intermediate(f'{self.selected_review_fname}.pkl',
                                selected_review)

    def prepare_summary(self, n_keywords: int = 4):
        """
        Now prepare data for summary page
        The output is [(topic_idx, [(word, weight, color_in_hex), ...]), ...]
        """

        # `temp_all_topic_word_weight_sent` = [(topic_idx, word, weight, sentiment), ...]
        temp_all_topic_word_weight_sent = []
        for topic in self.topic_words:
            temp_list = []
            for ww in topic[1]:
                if ww[0] in self.sent.keys():
                    temp_list.append(
                        (topic[0], ww[0], ww[1], self.sent[ww[0]]))
            temp_all_topic_word_weight_sent.extend(temp_list)
        # sort by sentiment
        temp_all_topic_word_weight_sent.sort(key=lambda x: x[3], reverse=True)

        # initialize topic_summary [(topic_idx, topic_sent_value, topic_keywords, [(word,weight,color_hex), ...]), ...]
        topic_summary = []
        for idx in range(self.n_topics):
            topic_summary.append([idx, None, None, []])

        for x in enumerate(temp_all_topic_word_weight_sent):
            color = list(COLORS.keys())[int(
                float(x[0]) / len(temp_all_topic_word_weight_sent) * 7.999)]
            topic_summary[x[1][0]][3].append((
                x[1][1],  # word
                np.float64(x[1][2]),  # weight
                color,  # sentiment color
            ))

        def get_topic_sent(t):
            d = {}
            for w in t:
                if w[2] in d.keys():
                    d[w[2]] += w[1]
                else:
                    d[w[2]] = w[1]
            return sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0]

        for topic in topic_summary:
            # topic_sent_value color string
            topic[1] = get_topic_sent(topic[3])
            # topic_keywords
            topic[2] = [
                kw[0] for kw in sorted(
                    topic[3], key=lambda x: x[1], reverse=True)[:n_keywords]
            ]

        # Save the data
        self._dump_intermediate(f"{self.topic_summary_fname}.pkl",
                                topic_summary)

    def run(self,
            n_topics: int = 4,
            gram_mode: str = 'bigram',
            probability_threshold: int = 0.3):
        self.read_file()
        self.preprocess()
        #self.parse_dependency()
        self.build_w2v_model()
        self.calc_senti_words()
        #self.map_dependency()
        self.topic_modeling(n_topics, gram_mode)
        self.prepare_review_list(probability_threshold)
        self.prepare_summary()
