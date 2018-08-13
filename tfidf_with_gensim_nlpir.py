from gensim import corpora
from gensim import models
import re
import json
import jieba
import datetime
import pynlpir
import datetime

def get_corpus():
    print("get corpos...")
    names = []
    contents = []
    path = "./data./bd_top3_random100000_sample.json"
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
                desc = desc + dict['name']
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

    path = "./output./100000_result_with_gensim_nlpirV1.txt"
    print("Starting Modeling...")
    dictionary = corpora.Dictionary(Corpus_list)
    #corpus = [dictionary.doc2bow(text) for text in Corpus_list]
    start = datetime.datetime.now()
    id2word = {}
    tfidf = models.TfidfModel(corpus = Corpus_list, id2word=id2word, dictionary=dictionary)
    end = datetime.datetime.now()
    print("Construct Model Time: %.5fs" % ((end - start).total_seconds()))
    print("Construct Model done!")
    #corpus_tfidf = tfidf[corpus]

    with open(path, 'w', encoding='utf-8') as f:
        print("Writing Data...")
        start = datetime.datetime.now()
        for i in range(len(names)):
            #print("Top words in", names[i])
            f.write("Top words in " + names[i] + "\n")
            test_corpus = dictionary.doc2bow(Corpus_list[i])
            test_corpus_tfidf = tfidf[test_corpus]
            test_corpus_tfidf = sorted(test_corpus_tfidf, key=lambda item: item[1], reverse=True)
            id2token = dict(zip(dictionary.token2id.values(), dictionary.token2id.keys()))
            length = len(test_corpus_tfidf)
            if length > 10:
                length = 10
            for j in range(length):
                word = id2token[test_corpus_tfidf[j][0]]
                score = test_corpus_tfidf[j][1]
                f.write("\tWord: " + word + ", TF-IDF: " + str(round(score, 5)) + "\n")
        end = datetime.datetime.now()
        print("Writing Time: %.5fs"%((end - start).total_seconds()))
    pynlpir.close()



if __name__ == "__main__":
    main()