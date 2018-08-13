import re
import os
import psutil
import json
import jieba
import pynlpir
import datetime
import datetime
import threading
from ctypes import c_char_p
from gensim import corpora
from gensim import models


def get_corpus():
    print("get corpos...")
    names = []
    contents = []
    business = []
    secondIndustry = []
    path = "./data./bd_top3_random10000_sample.json"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            dict = json.loads(line)
            if 'name' in dict.keys() and 'content' in dict.keys():
                names.append(dict['name'])
                desc = ""
                for item in dict['content']:
                    desc = desc + item['desc'] + "。"
                desc = desc.replace("[", "").replace("]", "").replace("\n", "。").replace("...", "。")
                desc = desc.replace(dict['name'], "")
                if  'business' in dict.keys():
                    desc = desc + dict['business']
                    business.append(dict['business'])
                else:
                    business.append(None)
                if 'secondIndustry' in dict.keys():
                    secondIndustry.append(dict['secondIndustry'])
                else:
                    secondIndustry.append(None)
                contents.append(desc)
    return names, contents, business, secondIndustry


#加载停用词表
def get_stopwords_and_types():

    stopwords = []
    stoptypes = []
    path = "./data./stop_words.txt"
    # path = "./data./stop_words.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    path = "./data./stop_words_type.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stoptypes.append(line.strip())
    return stopwords, stoptypes

def get_ind_business_kws():
    with open("./data/ind_business_kws.txt", "r", encoding='utf-8') as f:
        ind_business_kws = []
        lines = f.readlines()
        for line in lines:
            ind_business_kws.append(line.strip())
        return ind_business_kws

stop_words, stop_types = get_stopwords_and_types()
ind_business_kws = get_ind_business_kws()
pynlpir.open()
def add_User_Dictionary(ind_business_kws):
    print("add User Dictionary to pyNLPIR...")
    start = datetime.datetime.now()
    for word in ind_business_kws:
        if pynlpir.nlpir.AddUserWord(c_char_p(word.encode())) == 0:
            print(word, "error")

    end = datetime.datetime.now()
    print("add User Dictionary to pyNLPIR time: {0}s".format((end - start).total_seconds()))
add_User_Dictionary(ind_business_kws)
del ind_business_kws


#切词与过滤词预处理
def pre_process(text):

    text = text.lower()
    useful_word_pattern = u"^[a-zA-Z0-9\u4e00-\u9fa5]+$"
    district_pattern = u"[\u4e00-\u9fa5]{1,7}?(?:省|自治区|市|区|县|镇|村|街)$"
    tokens = pynlpir.segment(text, pos_names='child')
    filtered = [word[0] for word in tokens
                if 1 < len(word[0]) < 16 and not re.match(district_pattern, word[0]) and
                word[1] not in stop_types and word[0] not in stop_words and re.match(useful_word_pattern, word[0])]
    #print(test)
    return filtered

def process_corpus(corpus):
    print("cutting corpus...")
    start = datetime.datetime.now()
    corpus_list = []
    length = len(corpus)
    for i in range(length):
        corpus_list.append(pre_process(corpus[i]))
    end = datetime.datetime.now()
    print("cutting corpus done.")
    print("cutting time: {0}s".format((end - start).total_seconds()))
    return corpus_list

def getCorpus():
    name, contents, business, secondIndustry = get_corpus()
    Corpus_list = process_corpus(contents)
    return name, Corpus_list, business, secondIndustry

def main():

    print("Getting Corpus...")
    start = datetime.datetime.now()
    names, Corpus_list, business, secondIndustry = getCorpus()
    tfidfKeys = []
    end = datetime.datetime.now()
    print("Reading Data Time: %.5fs"%((end - start).total_seconds()))
    path = "./output./10000_result_with_gensim_nlpirV2.txt"
    print("Starting Modeling...")
    start = datetime.datetime.now()
    dictionary = corpora.Dictionary(Corpus_list)
    corpus = [dictionary.doc2bow(text) for text in Corpus_list]
    del Corpus_list
    id2token = dict(zip(dictionary.token2id.values(), dictionary.token2id.keys()))
    del dictionary
    tfidf = models.TfidfModel(corpus)
    end = datetime.datetime.now()
    print("Construct Model Time: %.5fs" % ((end - start).total_seconds()))
    print("Construct Model done!")

    with open(path, 'w', encoding='utf-8') as f:
        print("Writing Data...")
        start = datetime.datetime.now()
        for i in range(len(names)):
            f.write("Top words in " + names[i] + "\n")
            corpus_tfidf = tfidf[corpus[i]]
            corpus_tfidf = sorted(corpus_tfidf, key=lambda item: item[1], reverse=True)
            temp_key = []
            for id, score in corpus_tfidf:
                if score > 0:
                    f.write("\tWord: " + id2token[id] + ", TF-IDF: " + str(round(score, 5)) + "\n")
                    temp_key.append((id2token[id], round(score, 5)))
                else:
                    break
            tfidfKeys.append(temp_key)
        end = datetime.datetime.now()
        print("Writing Time: %.5fs"%((end - start).total_seconds()))

    with open("./output./10000_result_with_gensim_nlpir_jsonV2.json", 'w', encoding='utf-8') as f:
        for i in range(len(names)):
            jsondata = {}
            print(names[i], ":")
            print("business: ", business[i])
            print("SecondIndustry: ", secondIndustry[i])
            print("TopKeys:", tfidfKeys[i])
            jsondata['name'] = names[i]
            jsondata['business'] = business[i]
            jsondata['SecondIndustry'] = secondIndustry[i]
            jsondata['TopKeys'] = tfidfKeys[i]
            print(json.dumps(jsondata, ensure_ascii=False))
            f.write(json.dumps(jsondata, ensure_ascii=False))
            f.write('\n')
    pynlpir.close()



if __name__ == "__main__":
    main()