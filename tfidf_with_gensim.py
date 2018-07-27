from gensim import corpora
from gensim import models
import re
import json
import jieba
import datetime

def get_corpus():
    names = []
    contents = []
    path = "./data./bd_top3_sample_20180710.json"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            dict = json.loads(line)
            desc = ""
            temp_content = dict['content']
            if temp_content:
                for item in temp_content:
                    desc = desc + item['desc'] + "\n"
                contents.append(desc)
                names.append(dict['name'])
    return names, contents

def get_stopwords():

    stopwords = []
    path = "./data./stop_words.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

stopwords = get_stopwords()

def pre_process(text):

    tokens = jieba.lcut(text, HMM=True)
    striped = [word.strip() for word in tokens]
    number_pattern = "^\d*$"
    district_pattern = u'[\u4e00-\u9fa5]{1,7}?(?:省|自治区|市|区|县|镇|村|街|路)$'
    filtered = [word for word in striped if len(word) > 1 and not re.match(number_pattern, word)
                and not re.match(district_pattern, word) and word not in stopwords]
    return filtered


def getCorpus():
    name, contents = get_corpus()
    Corpus_list = []
    for i in range(len(contents)):
        Corpus_list.append(pre_process(contents[i]))
    return name, Corpus_list

def main():

    print("Getting Corpus...")
    start = datetime.datetime.now()
    names, Corpus_list = getCorpus()
    end = datetime.datetime.now()
    print("Reading Data Time: %.5fs"%((end - start).total_seconds()))

    path = "./output./2.8G_result_with_gensim.txt"
    print("Starting Modeling...")
    start = datetime.datetime.now()
    dictionary = corpora.Dictionary(Corpus_list)
    #corpus = [dictionary.doc2bow(text) for text in Corpus_list]
    id2word = {}
    tfidf = models.TfidfModel(corpus = Corpus_list, id2word=id2word, dictionary=dictionary)
    #corpus_tfidf = tfidf[corpus]

    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(names)):
            #print("Top words in", names[i])
            f.write("Top words in " + names[i] + "\n")
            test_corpus = dictionary.doc2bow(Corpus_list[i])
            test_corpus_tfidf = tfidf[test_corpus]
            test_corpus_tfidf = sorted(test_corpus_tfidf, key=lambda item: item[1], reverse=True)
            id2token = dict(zip(dictionary.token2id.values(), dictionary.token2id.keys()))
            for j in range(10):
                word = id2token[test_corpus_tfidf[j][0]]
                score = test_corpus_tfidf[j][1]
                #print("\tWord: %s, TF-IDF: %.5f" % (word, score))
                f.write("\tWord: " + word + ", TF-IDF: " + str(round(score, 5)) + "\n")
        end = datetime.datetime.now()
        print("Modeling Time: %.5fs"%((end - start).total_seconds()))



if __name__ == "__main__":
    main()