import logging
from gensim.models.phrases import Phrases
import os
import itertools
from extractSentenceWords import *

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)
#
# with open(self.file_path, 'r') as f:
#     rows = f.readlines()
#     line_count = 0
#     for r in rows:
#         row = r.split("******")
#         # filter unknown version
#         if row[4] != 'Unknown':
#             self.reviews.append(
#                 Review(row[0], row[1], row[2], row[3], row[4],
#                        row[5]))
#             line_count += 1
#     self.log.info(f"{line_count} reviews in total")


def build_input(app_files):
    doc_sent_word = []
    num_words = 0
    num_docs = 0

    l_id = 0
    app = "youtube"
    with open(app_files) as fin:
        for line in fin.readlines():
            line = line.strip()
            line = line.split("******")
            words_sents, wc = extractSentenceWords(line[1], lemma=True)
            doc_sent_word.append(words_sents)
            num_docs += 1
            num_words += wc
            if l_id % 1000 == 0:
                logging.info("processed %d docs of %s" % (l_id, app))
            l_id += 1
    logging.info("Read %d docs, %d words!" % (num_docs, num_words))
    return doc_sent_word


# def build_input(app_files):
#     doc_sent_word = []
#     num_words = 0
#     num_docs = 0
#     # for a in app_files:
#     #     print(a)
#     for app, path in app_files:
#         l_id = 0
#         with open(path) as fin:
#             for line in fin.readlines():
#                 line = line.strip()
#                 line = line.split("******")
#                 words_sents, wc = extractSentenceWords(line[1], lemma=True)
#                 doc_sent_word.append(words_sents)
#                 num_docs += 1
#                 num_words += wc
#                 if l_id % 1000 == 0:
#                     logging.info("processed %d docs of %s" % (l_id, app))
#                 l_id += 1
#     logging.info("Read %d docs, %d words!" % (num_docs, num_words))
#     return doc_sent_word


### write bigrams and trigrams to .model files
def extract_phrases(app_files, bigram_min, trigram_min):
    bigram_fp = os.path.join("model", "bigram.model")
    trigram_fp = os.path.join("model", "trigram.model")

    rst = build_input(app_files)
    gen = list(itertools.chain.from_iterable(rst))  # flatten
    bigram = Phrases(gen, threshold=5, min_count=bigram_min)
    trigram = Phrases(bigram[gen], threshold=3, min_count=trigram_min)
    # write
    bigram.save(bigram_fp)
    trigram.save(trigram_fp)
