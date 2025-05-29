#!/usr/bin/env python
# coding: utf-8

# # Preprocessing Stage
# ## استدعاء المكتبات
# هنا هنستدعي كل المكتبات اللي بنحتاجها خلال المشروع

# In[1]:


import warnings
warnings.filterwarnings("ignore")
from pyarabic import araby
import os
from nltk.tag import StanfordPOSTagger
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import re
from nltk import ISRIStemmer
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import time


# ## 1- Tokenization Function
# دي الفانكشن اللي بتقسم الجملة لكلمات <br>
# **اللي استدعيناها فوق araby لاحظ استخدام مكتبة**

# In[ ]:


def Tokening(sample):
    tokens = araby.tokenize(sample)
    return tokens


# 👇مثال: هنا مثلا بنديله جملة وهو بيقطعها ويسميها توكينز

# ## 2- Part Of Speech Handling
# دي الفانكشن اللي بتتعرف على اي اسم علم او حرف جر او كلمة غريبه وتحذفهم من الجملة وتخلي بس الصفات والافعال والاجزاء اللي تساعدنا في التقييم <br>
# **اللي استدعيناها فوق لاستدعاء الموديل المطلوب StanfordPOSTagger لاحظ استخدام مكتبة**

# In[ ]:


java_path = "C:/Program Files/Java/jre1.8.0_251/bin/java.exe"
os.environ['JAVAHOME'] = java_path


# In[ ]:


jar = "stanford-postagger-full-2018-10-16/stanford-postagger.jar"
model = "stanford-postagger-full-2018-10-16/models/arabic.tagger"
pos_tagger = StanfordPOSTagger(model, jar, encoding = 'utf8')


# In[ ]:


def PartOfSpeech(tokens):
    pos_words = pos_tagger.tag(tokens)
    filtered_tokens = []
    unwanted_tags = {"CC", "NNP", "PRP", 'CD', "IN", 'UH', 'DT'}
    for word in pos_words:
        if word[0]:
            if word[0].split('/')[1] not in unwanted_tags:
                filtered_tokens.append(word[0].split('/')[0])
        elif word[1].split('/')[1] not in unwanted_tags:
            filtered_tokens.append(word[1].split('/')[0])
    return filtered_tokens


# ## 3- Named Entity Recognization (NER) Handling
# دي الفانكشن اللي بتمسح اي زيادات من الكلمة علشان تكون كلها كلمات موحده ومفيش فرق في التشكيل <br>
# **اللي استدعيناها فوق لاستدعاء الموديل المطلوب transformers -> pipeline لاحظ استخدام مكتبة**

# In[2]:


tokenizer = AutoTokenizer.from_pretrained("models/NER")
model = AutoModelForTokenClassification.from_pretrained("models/NER")
Ner = pipeline("ner", model=model, tokenizer=tokenizer)


# In[ ]:


def NerDetective(sample):
    persons = []
    ner_obj = Ner(sample)
    unwanted_tags = {'B-PRICE', 'I-PRICE', "B-DISEASE", "I-DISEASE", 'B-PERSON', 'I-PERSON'}
    for i in range(len(ner_obj)):
        if ner_obj[i]["entity"] not in unwanted_tags:
            persons.append(ner_obj[i]["word"])
    return persons


# In[ ]:


def CleanNer(sample):
    words = []
    ner_detectived = NerDetective(sample)
    for word in ner_detectived:
        try:
            if word.startswith("##"):
                words[-1] = words[-1] + word.replace("##", "")
            else:
                words.append(word)
        except:
            pass
    return(words)


# In[ ]:


def RemoveNer(tokens, sample):
    cleaned_ner = CleanNer(sample)
    filltered_tokens = [token for token in tokens if token not in cleaned_ner]
    return filltered_tokens


# ## 4- Normalization Function
# دي الفانكشن اللي بتمسح اي زيادات من الكلمة علشان تكون كلها كلمات موحده ومفيش فرق في التشكيل <br>
# **اللي استدعيناها فوق re لاحظ استخدام مكتبة**

# In[ ]:


def Normalize(tokens):
    normalized_tokens = []
    for token in tokens:
        token = re.sub("[إأآا]", "ا", token)
        token = re.sub("ى", "ي", token)
        token = re.sub("ة", "ه", token)
        token = re.sub("[\W\da-zA-Z]", "", token)
        token = re.sub("_", " ", token)
        token = araby.strip_diacritics(token)
        token = araby.strip_tatweel(token)
        if token != "":
            normalized_tokens.append(token)
    return normalized_tokens


# ## 5- Removing Stop Words Function
# ### استدعاء الكلمات المستبعده
# هنا بناخد الكلمات الموجوده في الداتاسيت بتاعت الكلمات المستبعده (ستوب ووردس)

# In[ ]:


def StopWords():
    sample = open('DataSets\stop_words.txt', 'r', encoding='utf-8')
    sample_Words = str(sample.read())
    stopwords = Tokening(sample_Words)
    return stopwords


# ### مسح الكلمات المستبعده من الجملة المعطاه
# وهنا بنشوف لو الكلمة موجوده فيها نمسحها

# In[ ]:


def RemoveStopWords(tokens):
    stop_words = StopWords()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


# ## 6- Stemming Function
# هنا بنرجع الكلمة لأصلها <br>
# **اللي استدعيناها فوق ISRIStemmer لاحظ استخدام مكتبة**

# In[ ]:


def Stemming(tokens):
    stemmer = ISRIStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


# ## 7- Restore tokens to sentence function
# هنا بنرجع الكلمات الى جمل<br>

# In[ ]:


def ToSentence(tokens):
    sentence = ' '.join(token for token in tokens)
    return sentence


# ## 8- Generating Word Cloud Figure
# هنا بنعمل خريطة للكلمات الاكثر استخداماً في الجمل او المقال<br>

# In[ ]:


def wordcloud(df):
    text = " ".join(sentence for sentence in df['cleaned_text'])
    # Reshape and display the Arabic text
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    # Generate the word cloud
    wordcloud = WordCloud(font_path='tahomabd.ttf',
                          width=220, height=220,
                          margin=0, mode='RGBA', background_color=None).generate(bidi_text)
    # Plot the word cloud
    plt.figure(figsize=(3, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("assets/frame1/wordcloud.png", transparent = True)


# In[ ]:


def SentenceWordCloud(sentence):
    text = sentence
    # Reshape and display the Arabic text
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    # Generate the word cloud
    wordcloud = WordCloud(font_path='tahomabd.ttf',
                          width=1600, height=800,
                          margin=0).generate(bidi_text)
    # Plot the word cloud
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("assets/frame0/wordcloud.png", transparent = True)


# ### Cleaning & Preprocessing Functions

# In[ ]:


def SentencePreProcess(sentence):
    tokens = Tokening(sentence)
    tokens = Normalize(tokens)
    tokens = PartOfSpeech(tokens)
    sentence = ToSentence(tokens)
    tokens = RemoveNer(tokens, sentence)
    sentence = Stemming(tokens)
    sentence = RemoveStopWords(sentence)
    sentence = ToSentence(sentence)
    return sentence


# In[ ]:


def PreProcess(location):
    df = pd.read_csv(location, encoding="utf-8")
    df.columns = ['text']
    print("PreProcessing Started!")
    start_time = time.perf_counter()
    df['cleaned_text'] = df['text'].apply(Tokening)
    end_time = time.perf_counter()
    print("Tokenization complete!", end_time - start_time)
    start_time = time.perf_counter()
    df['cleaned_text'] = df['cleaned_text'].apply(Normalize)
    end_time = time.perf_counter()
    print("Normalization complete!", end_time - start_time)
    start_time = time.perf_counter()
    df['cleaned_text'] = df['cleaned_text'].apply(PartOfSpeech)
    end_time = time.perf_counter()
    print("POS filtering complete!", end_time - start_time)
    df['preprocessed_text'] = df['cleaned_text'].apply(ToSentence)
    start_time = time.perf_counter()
    df['cleaned_text'] = df.apply(lambda x: RemoveNer(x['cleaned_text'], x['preprocessed_text']), axis=1)
    end_time = time.perf_counter()
    print("NER filtering complete!", end_time - start_time)
    start_time = time.perf_counter()
    df['preprocessed_text'] = df['cleaned_text'].apply(Stemming)
    end_time = time.perf_counter()
    print("Stemming complete!", end_time - start_time)
    start_time = time.perf_counter()
    df['preprocessed_text'] = df['preprocessed_text'].apply(RemoveStopWords)
    end_time = time.perf_counter()
    print("Stopword removal complete!", end_time - start_time)
    df['cleaned_text'] = df['cleaned_text'].apply(ToSentence)
    df['preprocessed_text'] = df['preprocessed_text'].apply(ToSentence)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

