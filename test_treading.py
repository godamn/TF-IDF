import re
import os
import psutil
import json
import jieba
import datetime
import pynlpir
import datetime
import threading
import multiprocessing
from multiprocessing import Pool
from gensim import corpora
from gensim import models


def get_corpus():
    print("get corpos...")
    names = []
    contents = []
    path = "./data./bd_top3_random1000_sample.json"
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
                contents.append(desc)
    return names, contents


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

stop_words, stop_types = get_stopwords_and_types()
name, corpus = get_corpus()
length = len(corpus)
# corpus_list = [None for x in range(length)]

pynlpir.open()

#切词与过滤词预处理
def pre_process(text):
    text = text.lower()
    useful_word_pattern = u"^[a-zA-Z0-9\u4e00-\u9fa5]+$"
    district_pattern = u"[\u4e00-\u9fa5]{1,7}?(?:省|自治区|市|区|县|镇|村|街)$"
    tokens = pynlpir.segment(text, pos_names='child')
    filtered = [word[0] for word in tokens
                if len(word[0]) > 1 and len(word[0]) < 9 and not re.match(district_pattern, word[0]) and
                word[1] not in stop_types and word[0] not in stop_words and re.match(useful_word_pattern, word[0])]
    #print(test)
    return filtered



def process_thread(i, corpus_list):
    while i < length:
        filtered = pre_process(corpus[i])
        corpus_list[i] = filtered
        i = i + 4


def main():
    corpus_list = multiprocessing.Manager().list(range(length))
    # threads = []
    # print("cutting corpus...")
    #
    # for i in range(num_thread):
    #     t = threading.Thread(target=process_thread, args=(i,))
    #     threads.append(t)
    #     t.start()
    #
    # for t in threads:
    #     t.join()
    # print("cutting corpus done.")
    start = datetime.datetime.now()
    p = Pool()
    for i in range(4):
        p.apply_async(process_thread, args=(i,corpus_list))
    p.close()
    p.join()
    pynlpir.close()
    end = datetime.datetime.now()
    print("Time: %.5fs" % ((end - start).total_seconds()))
    path = "./output./10000_result_with_gensim_nlpirV4.txt"
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
            for id, score in corpus_tfidf[:10]:
                f.write("\tWord: " + id2token[id] + ", TF-IDF: " + str(round(score, 5)) + "\n")
        end = datetime.datetime.now()
        print("Writing Time: %.5fs"%((end - start).total_seconds()))


if __name__ == "__main__":
    main()