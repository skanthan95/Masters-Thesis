import requests 
from bs4 import BeautifulSoup 
import pandas as pd 
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import docx 
import pprint
import nltk 
from nltk.tokenize import word_tokenize
import spacy
import re 
import urllib.parse 
from urllib.parse import urljoin
import io 
import json 
import os.path 
import os 
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import unicodedata
import en_core_web_sm
nlp = en_core_web_sm.load()
import string
import gensim

# SECTION 1: Scraping and cleaning the texts for analysis, saving as dataframes

def remove_punct(column, df):
  '''
  Helper function for scrape_mb; ensures
  that no punctuation slipped through the cracks, 
  all removed for analysis
  '''
  # define punctuation
  punctuations = '''"--"!()[]{};:'"\,<>./?@#$%^&*_~'''

  # remove punctuation from the string
  no_punct = ""

  for row in df[column]: # looking at each row
      for char in row: # looking at each word in each row
          if char not in punctuations:
              no_punct = no_punct + char
              row = no_punct 

  return df

def scrape_il(link):
  '''
  Takes base link and finds all hyperlinks within it;
  Takes those and constructs the links for each chapter
  in the Iliad. Then, scrapes and cleans the text and 
  returns a df of the text and corresponding chapter
  '''
  req = Request(str(link))
  html_page = urlopen(req)
  soup = BeautifulSoup(html_page, 'lxml')
  links = []
  for link in soup.findAll('a'):
    links.append(link.get('href'))
  
  sub_links = []
  for href in links:
      base = 'http://www.gutenberg.org/files/2199/2199-h/2199-h.htm'
      join = urljoin(base, href)
      sub_links.append(join)
  
  sub_links = [link for link in sub_links if "chap" in link]
    
  sub_texts = []
  for link in sub_links:
      URL = link
      page = requests.get(URL)
      txt = page.text
      mbSoup = BeautifulSoup(txt, 'html.parser')
      soup_text = mbSoup.text
      text_strip = soup_text.strip()
      newline = text_strip.strip('\n')
      strplace = newline.replace('\n','')
      nums = re.sub('[^A-Za-z0-9]+', ' ', strplace)
      normalize = unicodedata.normalize("NFKD", nums)
      lower = normalize.lower()
      
      data = {'URL': URL, 'Text': lower}
      sub_texts.append(data)

  il_df = pd.DataFrame(sub_texts)
  
  return il_df 

def scrape_mb(main_link, sub_links):
  '''
  Takes base link and records all "Section" links.
  Takes list of Mahabharata book links and scrapes
  text from each recorded section, cleans it, and
  returns df of text and corresponding section & chapter
  '''
  req = Request(str(main_link))
  html_page = urlopen(req)
  soup = BeautifulSoup(html_page, "lxml")
  output_links = []
  for tag in soup.findAll('a'):
      if tag.text.startswith("Section"):
          output_links.append(tag.get('href'))
  output_links = [[] for i in range(len(sub_links))]
  fin_links = [[] for i in range(len(sub_links))] 
  fin_links_text = [[] for i in range(len(sub_links))]  

  i = 0
  for link in sub_links:
      url = requests.get(link)
      data = url.text
      soup = BeautifulSoup(data)
      for tag in soup.findAll('a'): # searching for 'a' tag (corresponds to link)
          if tag.text.startswith("Section") or tag.text.startswith("section"):
              output_links[i].append(tag.get('href')) # goal is to get list of sublinks in links list 
                                        #(ideally, one links list per book link, so 18 lists)
      i += 1
          
  i = 0
  for l in output_links:
      for href in l:
          full = "https://www.sacred-texts.com/hin/" + str(href[0:3]) + "/" + href
          fin_links[i].append(full)
          texturl = requests.get(full)
          textdata = texturl.text
          textsoup = BeautifulSoup(textdata)
          tag_text_all = ''
          for tag in textsoup.findAll('p'): # searching for 'p' tag (corresponds to text paragraphs)
              tag_text_all = tag_text_all + ' ' + tag.text
          fin_links_text[i].append(tag_text_all)
          
      i += 1

  flat_links = sum(fin_links, [])
  flat_links_text = sum(fin_links_text, [])
  mb = pd.DataFrame(flat_links, columns = ['Section link'])  
  mb['Text'] = flat_links_text
  mb['cleaned_text'] = mb['Text'].str.lower()
  mb['cleaned_text'] = mb['cleaned_text'].str.strip()
  mb['cleaned_text'] = mb['cleaned_text'].str.strip('\n')
  mb['cleaned_text'] = mb['cleaned_text'].str.replace('\n','')
  mb['cleaned_text'] = mb['cleaned_text'].apply(lambda text: unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii'))
  mb['cleaned_text'] = mb['cleaned_text'].str.replace('\d+', '')

  
  mb_df = remove_punct('cleaned_text', mb)

  return mb_df

  il_text = scrape_il("http://www.gutenberg.org/files/2199/2199-h/2199-h.htm#chap01")
  mb_full = scrape_mb("https://www.sacred-texts.com/hin/m06/index.htm", 
  ["https://www.sacred-texts.com/hin/m01/index.htm",
                "https://www.sacred-texts.com/hin/m02/index.htm",
                "https://www.sacred-texts.com/hin/m03/index.htm",
                "https://www.sacred-texts.com/hin/m04/index.htm",
                "https://www.sacred-texts.com/hin/m05/index.htm",
                "https://www.sacred-texts.com/hin/m06/index.htm",
                "https://www.sacred-texts.com/hin/m07/index.htm",
                "https://www.sacred-texts.com/hin/m08/index.htm",
                "https://www.sacred-texts.com/hin/m09/index.htm",
                "https://www.sacred-texts.com/hin/m10/index.htm",
                "https://www.sacred-texts.com/hin/m11/index.htm",
                "https://www.sacred-texts.com/hin/m12/index.htm",
                "https://www.sacred-texts.com/hin/m13/index.htm",
                "https://www.sacred-texts.com/hin/m14/index.htm",
                "https://www.sacred-texts.com/hin/m15/index.htm",
                "https://www.sacred-texts.com/hin/m16/index.htm",
                "https://www.sacred-texts.com/hin/m17/index.htm",
                "https://www.sacred-texts.com/hin/m18/index.htm"])

# SECTION 2: Word2vec, Word concordance adjective plots, BookNLP

# Custom stop-words list
stop_words = ["p", ".", "i", "me", "my", "myself", "we", "our", "ours", 
              "ourselves", "you", "your", "yours", "yourself", "yourselves", 
              "he", "him", "his", "himself", "she", "her", "hers", "herself", 
              "it", "its", "itself", "they", "them", "their", "theirs", 
              "themselves", "what", "which", "who", "whom", "this", "that", 
              "these", "those", "am", "is", "are", "was", "were", "be", "been",
              "being", "have", "has", "had", "having", "do", "does", "did", 
              "doing", "a", "an", "the", "and", "but", "if", "or", "because", 
              "as", "until", "while", "of", "at", "by", "for", "with", "about", 
              "against", "between", "into", "through", "during", "before", 
              "after", "above", "below", "to", "from", "up", "down", "in", 
              "out", "on", "off", "over", "under", "again", "further", "then", 
              "once", "here", "there", "when", "where", "why", "how", "all", 
              "any", "both", "each", "few", "more", "most", "other", "some", 
              "such", "no", "nor", "not", "only", "own", "same", "so", "than", 
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def word_tokenize(word_list):
  '''
  Tokenizes data
  '''
  tokenized = []
  doc = nlp(word_list)
  for token in doc:
      if not token.is_punct and len(token.text.strip()) > 0:
          tokenized.append(token.text)
  return tokenized

def w2v_ready(sentences):   
  '''
  Removes stopwords, making data ready for word2vec model;
  digits and other unnecessary characters removed in Section 1
  ''' 
  model_ready = []
  for sent in sentences:
    words = word_tokenize(sent)
    tokens_without_sw = [word for word in words if not word in stop_words]
    model_ready.append(tokens_without_sw)
  
  return model_ready


model_ready_il = w2v_ready(il_df['Text'].to_list())
il_w2v = gensim.models.word2vec.Word2Vec(model_ready_il)

model_ready_mb = w2v_ready(mb_df['cleaned_text'].to_list())
mb_w2v = gensim.models.word2vec.Word2Vec(model_ready_mb)


# Finding the words most similar to keywords (examples)

il_w2v.most_similar('car')

#[('woman', 0.611786961555481),
# ('prayer', 0.6010794043540955),
# ('honour', 0.5999828577041626),
# ('full', 0.5984799861907959),
# ('camp', 0.5983457565307617),
# ('daughter', 0.5970908403396606),
# ('keep', 0.5964771509170532),
# ('great', 0.5908088684082031),
# ('arose', 0.5900729894638062),
# ('word', 0.5884246826171875)]


# Token files generated by BookNLP, which included part of speech tags
# (stanford tagger didn't accurately tag adjectives)

rig_tokens = pd.read_csv('rig.txt.tokens.csv')
theo_tokens = pd.read_csv('theo.txt.tokens.csv')
il_tokens = pd.read_csv('il.txt.tokens.csv')
mb_tokens = pd.read_csv('mb.txt.tokens.csv')
shield_tokens = pd.read_csv('heracles.txt.tokens.csv')


def word_plot(adj_list):
  '''
  Plots frequency plots for each list of adjectives
  '''
    word_list = adj_list
    counts = Counter(word_list)
    labels, values = zip(*counts.items())
    # sort your values in descending order
    indSort = np.argsort(values)[::-1]
    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))
    bar_width = 0.35
    plt.bar(indexes, values)
    # add labels
    plt.xticks(indexes + bar_width, labels)
    plt.xticks(rotation=90)
    plt.show()

def preceding_word(data, keyword_list):
  '''
  Compiles words preceding keywords of interest in keyword_list
  and performs NER on them. Then calculates proportion of adjectives for each keyword
  (compared to other parts of speech),  as well as the numerator and denominator for that proportion
  '''
    data = data[data['normalizedWord'].str.contains('[A-Za-z]', na=False)]  
    data['tupes'] = list(zip(data.originalWord, data.pos))
    col_tupes = data['tupes'].tolist()
    lst = []
    for index, tup in enumerate(col_tupes):
        if tup[0] in keyword_list:
            lst.append(col_tupes[index-1])
            
    # Named Entity Recognition to make more sense of what sorts of words precede these keywords
    ner_words = [nlp(tup[0]) for tup in lst]
    ner_tags = [word.label_) for word in ner_words]
    print(Counter(ner_tags))
    
    adj = []
    for tup in lst:
        if tup[1] == 'JJ':
            adj.append(tup[0])
    print("\n The proportion of adjectives for", keyword_list[0],"is", len(adj)/len(lst))
    print("The total number of preceding words is:", len(lst))
    print("The total number of preceding adjectives is:", len(adj))
    print("all the words:", Counter(lst))
    plot = word_plot(adj[:50])
    print(Counter(adj))

# keyword lists:
arrows = ['arrow', 'arrows', 'Arrow', 'Arrows']
chariots = ['chariot', 'chariots', 'Chariot', 'Chariots']
spears = ['spear', 'spears', 'Spear', 'Spears']
cars = ['car', 'cars', 'Car', 'Cars']
shafts = ['shaft', 'shafts', 'Shaft', 'Shafts']
arrows_and_shafts = arrows+shafts
chariots_and_cars = cars+chariots

## call preceding_word() on each text's list of tokens for each keyword;
## eg. preceding_word(il_tokens, spears)

# SECTION 3: SentenceBERT

# the control texts
hera = pd.read_csv('shield_heracles.csv')
theo = pd.read_csv('1final_theo.csv')
rig =  pd.read_csv('1final_rig.csv')

def text_clean(df):
    '''
    Cleans texts, uses sentence tokenizer to create sentences from data
    Returns list of sentences for dataframe
    '''
    df['cleaned_text'] = df['cleaned_text'].str.strip('\n')
    df['cleaned_text'] = df['cleaned_text'].str.strip('\n\n')
    df['cleaned_text'] = df['cleaned_text'].apply(lambda text: unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii'))
    df['cleaned_text'] = df['cleaned_text'].str.replace('\d+', '')
    df['cleaned_text'] = df['cleaned_text'].str.replace('\n', '')
    df['cleaned_text'] = df['cleaned_text'].str.replace(' \n', '')


    df['cleaned_text'] = df['cleaned_text'].str.replace('paragraph continues', '')
    df = df.replace(r'\\n',' ', regex=True) 


    for char in spec_chars:
        df['cleaned_text'] = df['cleaned_text'].str.replace(char, '')

    df['cleaned_text'] = df['cleaned_text'].str.replace('     ', ' ')
    df['cleaned_text'] = df['cleaned_text'].str.replace('  ', ' ')

    doc_df = ' '.join(df['cleaned_text'].tolist())
    
    sent_list = nltk.tokenize.sent_tokenize(doc_df)

    return sent_list

hera_sents = text_clean(hera)

# Filtering just the battle books
mb_not_b = [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18]
il_not_b = [1, 2, 4, 9, 18, 19, 22, 23, 24]

mb_battle = mb[~mb['book_num'].isin(mb_not_b)]
il_battle = il[~il['book_num'].isin(il_not_b)]
mb_bsents = text_clean(mb_battle)
il_bsents = text_clean(il_battle)
mb_sents = text_clean(mb)
il_sents = text_clean(il)
theo_sents = text_clean(theo)
rig_sents = text_clean(rig)

def tokenize(sentences):    
    tokenized_sent = []
    for s in sentences:
        tokenized_sent.append(word_tokenize(s.lower()))
    return tokenized_sent

tokenized_mb = tokenize(mb_sents)
tokenized_il = tokenize(il_sents)
tokenized_theo = tokenize(theo_sents)
tokenized_rig = tokenize(rig_sents)
tokenized_mbb = tokenize(mb_bsents)
tokenized_ilb = tokenize(il_bsents)


war_keywords = ['arm', 'forehead', 'trunk', 'sever', 'cut', 'blood', 'spurt', 'gush', 'slash', 'shaft', 'car', 'arrow', 
                'spear', 'chakra', 'sanguine', 'leg', 'thigh', 'groin', 'shoulder', 'stomach', 'pierce',
                'stab', 'chest', 'breast', 'torso', 'bleed', 'kill', 'die', 'perish', 'throat', 'moksha', 'bled']

war_sents_mb = []

for sent in mb_bsents:
    words = sent.split()
    for word in words:
        if word in war_keywords:
            war_sents_mb.append(sent)

war_sents_il = []            

for sent in il_bsents:
    words = sent.split()
    for word in words:
        if word in war_keywords:
            war_sents_il.append(sent)

war_sents_rig = []

for sent in rig_sents:
    words = sent.split()
    for word in words:
        if word in war_keywords:
            war_sents_rig.append(sent)

war_sents_theo = []            

for sent in theo_sents:
    words = sent.split()
    for word in words:
        if word in war_keywords:
            war_sents_theo.append(sent)

war_sents_hera = []            

for sent in hera_sents:
    words = sent.split()
    for word in words:
        if word in war_keywords:
            war_sents_hera.append(sent)

war_sents_hesiod = war_sents_theo + war_sents_hera

#comp_war = compare_texts(war_sents_mb, war_sents_il) # run this code tonight
#comp_war

sun_rituals = ['dawn', 'rising sun', 'sunrise', 'sunset', 'sun set', 'dusk']

sun_sents_mb = []

for sent in mb_bsents:
    words = sent.split()
    for word in words:
        if word in sun_rituals:
            sun_sents_mb.append(sent)

sun_sents_il = []            

for sent in il_bsents:
    words = sent.split()
    for word in words:
        if word in sun_rituals:
            sun_sents_il.append(sent)
            
sun_sents_rig = []

for sent in rig_sents:
    words = sent.split()
    for word in words:
        if word in sun_rituals:
            sun_sents_rig.append(sent)

sun_sents_theo = []            

for sent in theo_sents:
    words = sent.split()
    for word in words:
        if word in sun_rituals:
            sun_sents_theo.append(sent)

sun_sents_hera = []            

for sent in hera_sents:
    words = sent.split()
    for word in words:
        if word in sun_rituals:
            sun_sents_hera.append(sent)



foot_sents_mb = [sent for sent in mb_bsents if "foot" in sent or "sole" in sent or "feet" in sent or "heel" in sent or "ankle" in sent]
foot_sents_il = [sent for sent in il_bsents if "foot" in sent or "sole" in sent or "feet" in sent or "heel" in sent or "ankle" in sent]
foot_sents_rig = [sent for sent in rig_sents if "foot" in sent or "sole" in sent or "feet" in sent or "heel" in sent or "ankle" in sent]
foot_sents_theo = [sent for sent in theo_sents if "foot" in sent or "sole" in sent or "feet" in sent or "heel" in sent or "ankle" in sent]
foot_sents_hera = [sent for sent in hera_sents if "foot" in sent or "sole" in sent or "feet" in sent or "heel" in sent or "ankle" in sent]
foot_sents_hesiod = foot_sents_hera + foot_sents_theo

def compare_texts(sentlist1, sentlist2):
  '''
  Pairwise comparison of all sentences in specific keyword-topic-list
  (epic vs. epic), (epic vs. control)...
  '''
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # Two lists of sentences
    sentences1 = sentlist1 # the doc_mb sentences after being passed into the sentence transformer

    sentences2 = sentlist2 # the doc_il sentences after being passed into the sentence transformer

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)



    #Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j].item()})


    print(len(pairs))

    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    for pair in pairs[0:300]:
        i, j = pair['index']
        print("Sent1:  {} \n Sent2:  {} \n Score: {:.4f}".format(sentences1[i], sentences2[j], pair['score']))
    return [d['score'] for d in pairs]


# Getting pairwise similarity scores for all keyword-sentence occurences between
# the text pairs

sun_epics = compare_texts(sun_sents_mb, sun_sents_il) 
injury_epics = compare_texts(war_sents_mb, war_sents_il)
foot_epics = compare_texts(foot_sents_mb, foot_sents_il)

# repeat with all combinations of texts

data = [['mb_rig_sun_scores', mb_rig_sun],
        ['mb_hesiod_sun_scores', mb_hesiod_sun],
       ['il_rig_sun_scores', il_rig_sun],
       ['il_hesiod_sun_scores', il_hesiod_sun],
        ['mb_il_sun_scores', sun_epics],
       ['mb_rig_war_scores', mb_rig_war],
        ['mb_hesiod_war_scores', mb_hesiod_war],
       ['il_rig_war_scores', il_rig_war],
        ['il_hesiod_war_scores', il_hesiod_war],
        ['mb_il_war_scores', injury_epics],
       ['mb_rig_foot_scores', mb_rig_foot],
        ['mb_hesiod_foot_scores', mb_hesiod_foot],
       ['il_rig_foot_scores', il_rig_foot],
        ['il_hesiod_foot_scores', il_hesiod_foot],
       ['mb_il_foot_scores', foot_epics],
       ['hesiod_rig_sun_scores', hesiod_rig_sun],
       ['hesiod_rig_war_scores', hesiod_rig_war],
       ['hesiod_rig_foot_scores', hesiod_rig_foot]] 

df = pd.DataFrame(data, columns = ['text_combo', 'cos_scores']) 
co_vals = df['cos_scores'].to_list()

names = df['text_combo'].to_list()

def takeThird(elem):
    return elem[3]

# Descriptive stats for similarity scores 

avg_list = []

for ind,val in enumerate(co_vals):
    avg_list.append(("This is the average cos_sim for", names[ind], ":", (sum(val)/len(val))))

sorted(avg_list, key=takeThird, reverse=True)

max_list = []
for ind,val in enumerate(co_vals):
    max_list.append(("This is the max cos_sim for", names[ind], ":", max(val)))
    
desc_order = list(sorted(max_list, key=takeThird, reverse=True))
for item in desc_order:
    print(item, "\n")

range_list = []
for ind,val in enumerate(co_vals):
    range_list.append(("This is the range of cos_sim for", names[ind], ":", (max(val)-min(val))))
desc_range = list(sorted(range_list, key=takeThird, reverse=True))

for item in desc_range:
    print(item, "\n")

