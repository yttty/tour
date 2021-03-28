from flask import Flask, Response, flash, render_template, request, send_from_directory, redirect
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import pickle
import hashlib
import json
from tossa import TopicSentimentAnalysis
from logger import Logger
from config import COLORS, UPLOAD_DIR, Review
from main_run import main_run
from get_input import get_input

name = "ToSSA Server"
app = Flask(name)
logger = Logger(name)

progress_percentage = 0
tossa = TopicSentimentAnalysis(app.root_path, '')


def load_intermediate(rootpath, fname):
    with open(os.path.join(rootpath, "intermediate_result", fname), 'rb') as f:
        return pickle.load(f)


@app.route('/')
def render_index():
    return render_template('index.html')


@app.route('/progress')
def progress():
    def generate():
        global progress_percentage
        return "data:" + str(progress_percentage) + "\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    sample_folder = os.path.join(app.root_path, "example")
    return send_from_directory(directory=sample_folder, filename=filename)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == "":
            return "Please select a file!"

        if not os.path.isdir(UPLOAD_DIR):
            os.mkdir(UPLOAD_DIR)

        unique_filename = datetime.now().strftime(
            "%d-%b-%Y %H_%M_%S") + "_" + f.filename
        md5_filename = hashlib.md5(
            unique_filename.encode(encoding='UTF-8')).hexdigest()
        file_path = os.path.join(app.root_path, UPLOAD_DIR,
                                 secure_filename(md5_filename))
        f.save(file_path)

        logger.info(f'Upload file {md5_filename} successfully')

        # idea store  now can't save
        # basepath = os.path.dirname(__file__)
        # upload_path = os.path.join(basepath, 'dataset\youtube',
        #                            secure_filename(f.filename))

        global tossa
        tossa = TopicSentimentAnalysis(app.root_path, file_path)
        logger.info('Processing reviews...')
        tossa.read_file()

        return redirect(f"/parameter?fn={md5_filename}")


@app.route('/parameter', methods=['GET', 'POST'])
def parameter():
    global progress_percentage
    if request.method == 'GET':
        try:
            fn = request.args.get("fn")
        except TypeError:
            return "Invalid URL!"
        return render_template('parameter.html',
                               fn=fn,
                               version=tossa.get_version())
    if request.method == 'POST':
        progress_percentage = 0
        #read
        with open("static/pwords.txt", "r") as f:
            pwords = f.read().splitlines()
        with open("static/nwords.txt", "r") as f:
            nwords = f.read().splitlines()
        try:
            n_topics = int(request.values.to_dict().get('n_topics'))
            probability_threshold = float(
                request.values.to_dict().get('probability_threshold')) / 100
            #add
            win_size = int(request.values.to_dict().get('win_size'))
            bigram_min = int(request.values.to_dict().get('bigram_min'))
            trigram_min = int(request.values.to_dict().get('trigram_min'))

            assert n_topics > 0 and 0 < probability_threshold < 1 and win_size > 0 and bigram_min > 0 and trigram_min > 0
            fn = request.args.get("fn")
            pwords = list(
                set(request.values.to_dict().get('pwords').split(';') +
                    pwords))
            nwords = list(
                set(request.values.to_dict().get('nwords').split(';') +
                    nwords))
        except:
            return "Please input valid parameters!"

        logger.info(f'filename: {fn}; PosWords: {pwords}; NegWords: {nwords};')

        # analyze and generate files for rendering
        tossa.preprocess(version=None)
        progress_percentage += 10

        #tossa.parse_dependency()
        progress_percentage += 10

        tossa.build_w2v_model()
        progress_percentage += 10

        tossa.calc_senti_words(pwords, nwords)
        progress_percentage += 10

        #tossa.map_dependency()
        tossa.topic_modeling(n_topics, 'bigram')
        progress_percentage += 20

        tossa.prepare_review_list(probability_threshold)
        progress_percentage += 10
        # add
        main_run(n_topics, win_size, bigram_min, trigram_min, fn)
        progress_percentage += 10
        # cmd = "python get_input.py result/youtube " + str(n_topics)
        # os.system(cmd)
        get_input(n_topics, fn)
        progress_percentage += 10

        tossa.prepare_summary()
        progress_percentage += 10

        logger.info('Finished processing. Redirecting to summary page...')

        return redirect(f"/summary?fn={fn}")


@app.route('/summary', methods=['GET', 'POST'])
def render_summary():
    """Show topic's word cloud and sentiment in one page"""
    try:
        fn = request.args.get("fn")
    except:
        return "Invalid parameters!"

    coherence_value = load_intermediate(app.root_path, "coherence_value.pkl")
    summary = load_intermediate(app.root_path, "topic_summary.pkl")
    topic_sent = [(topic[1], COLORS[topic[1]]) for topic in summary]
    topic_keywords = [' '.join(topic[2]) for topic in summary]
    summary_json = json.dumps(summary)
    topic_sent_json = json.dumps(topic_sent)
    topic_keywords_json = json.dumps(topic_keywords)
    return render_template('summary.html',
                           fn=fn,
                           summary=summary_json,
                           coherence_value=coherence_value,
                           topic_sent=topic_sent_json,
                           topic_keywords=topic_keywords_json)


@app.route('/topic', methods=['GET'])
def render_topic_review_detail():
    """Show topic's detailed review"""
    try:
        topic_id = int(request.args.get("id"))
        version = request.args.get("ver")
    except TypeError:
        return "Missing topic id!"
    word_color = dict(
        map(lambda x: (x[0], x[2]),
            load_intermediate('.', "topic_summary.pkl")[topic_id][3]))
    selected_review = load_intermediate(app.root_path, "selected_review.pkl")

    def colorize_helper(token):
        if token in word_color.keys():
            return f" <strong style=\"color:  {word_color[token]} \"> {token} </strong> "
        else:
            return token

    def colorize(review_text: str):
        return ' '.join(map(colorize_helper, review_text.split()))

    review = [(review[2].rating, colorize(review[2].body), review[2].date,
               review[2].version, review[1]) for topic in selected_review
              for review in topic
              if review[0] == topic_id and review[2].version == version]
    return render_template('topic.html', topic_id=topic_id, review=review)


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
