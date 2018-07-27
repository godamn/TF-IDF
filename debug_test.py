import re
import json
import pynlpir
import datetime
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

#读取数据构建语料库
def get_corpus():
    print("get corpos...")
    names = []
    contents = []
    path = "./data./test.json"
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        dict = json.loads(line)
        names.append(dict['name'])
        desc = ""
        for item in dict['content']:
            desc = desc + item['desc'] + "\n"
        print(desc)
        contents.append(desc)
    return names, contents


#加载停用词表
def get_stopwords():

    stopwords = []
    path = "./data./stop_words_thulac.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

stop_words = get_stopwords()
pynlpir.open()
stop_wordtype = ['time word', 'personal name', 'toponym','noun of locality', 'adjective', 'numeral', 'pronoun',
                 'Chinese given name', 'transcribed personal name','preposition']

#切词与过滤词预处理
def pre_process(text):

    text_ = re.sub("[\s+\.\!\/_,^*(+\"\')]+|[+——()?【】“”！，。？、~…*（）]+", "", text)#过滤标点符号
    tokens = pynlpir.segment(text_, pos_names='child')
    filtered = [word[0] for word in tokens if len(word[0]) > 1 and word[1] not in stop_wordtype and word[0] not in stop_words]
    print(filtered)
    return filtered

#构建TfidfVectorizer模型
def tfidf(corpus):

    vectorizer = TfidfVectorizer(analyzer=pre_process)
    tf_idf = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    weights = tf_idf.tolil()#稀疏矩阵转换格式加速读取
    return words, weights


def main():
    path = "./output./10000_result_with_pynlpir.txt"
    with open(path, 'w', encoding='utf-8') as f:
        print("Getting Corpus...")
        start = datetime.datetime.now()
        names, contents = get_corpus()
        # for i in range(len(names)):
        #     print(names[i], ": ")
        #     pre_process(contents[i])

        end = datetime.datetime.now()
        print("Reading Data Time: %.5fs" % ((end - start).total_seconds()))
        f.write("Reading Data Time: %.5fs" % ((end - start).total_seconds())+'\n')
        print("finish getting corpos...\nstart modeling...")

        start = datetime.datetime.now()
        words, weights = tfidf(contents)
        print("finish modeling...")
        print(weights.shape)
        f.write("weights.shape is %s"%(str(weights.shape))+'\n')
        end = datetime.datetime.now()
        print("Modeling Time: %.5fs" % ((end - start).total_seconds()))
        f.write("Modeling Time: %.5fs" % ((end - start).total_seconds()) + '\n')
        pynlpir.close()

if __name__ == "__main__":
    main()
