from flask import Flask,url_for,request,render_template,jsonify,send_file
import json

# NLP Pkgs
import spacy
from textblob import TextBlob 
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')


# WordCloud & Matplotlib Packages
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from io import BytesIO
import random
import time


# initialize app
app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	# Receives the input query from form
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		# Analysis
		docx = nlp(rawtext)
		# Tokens
		custom_tokens = [token.text for token in docx ]
		# Word Info
		custom_wordinfo = [(token.text,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
		custom_postagging = [(word.text,word.tag_,word.pos_,word.dep_) for word in docx]
		# NER
		custom_namedentities = [(entity.text,entity.label_)for entity in docx.ents]


		blob = TextBlob(rawtext)
		blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
		# allData = ['Token:{},Tag:{},POS:{},Dependency:{},Lemma:{},Shape:{},Alpha:{},IsStopword:{}'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
		allData = [('"Token":"{}","Tag":"{}","POS":"{}","Dependency":"{}","Lemma":"{}","Shape":"{}","Alpha":"{}","IsStopword":"{}"'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop)) for token in docx ]

		result_json = json.dumps(allData, sort_keys = False, indent = 2)

		end = time.time()
		final_time = end-start

	return render_template('index.html',ctext=rawtext,custom_tokens=custom_tokens,custom_postagging=custom_postagging,custom_namedentities=custom_namedentities,custom_wordinfo=custom_wordinfo,blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,final_time=final_time,result_json=result_json)
	

# IMAGE WORDCLOUD
@app.route('/images')
def imagescloud():
    return "<h2>Enter text into url eg. /fig/yourtext</h2>"


@app.route('/images/<mytext>')
def images(mytext):
    return render_template("index.html", title=mytext)

@app.route('/fig/<string:mytext>')
def fig(mytext):
    plt.figure(figsize=(20,10))
    wordcloud = WordCloud(background_color='white', mode = "RGB", width = 1200, height = 1000).generate(mytext)
    plt.imshow(wordcloud)
    plt.axis("off")
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
	app.run(debug=True)