import json
import re
import time
import datetime
import jieba.analyse
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

#语料库构建
corpus = []

# 创建停用词list
def get_stopwords():
    stopwords = []
    path = "./data./stop_words.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


# 对句子进行分词
def pre_process(text):

    tokens = jieba.lcut(text)
    striped = [word.strip() for word in tokens]
    number_pattern = "^\d*$"
    district_pattern = u'[\u4e00-\u9fa5]{1,7}?(?:省|自治区|市|区|县|镇|村|街|路)$'
    stopwords = get_stopwords()
    filtered = [word for word in striped if len(word) > 1 and not re.match(number_pattern, word)
                and not re.match(district_pattern, word) and word not in stopwords]
    return filtered

#读取数据
def construct_corpus():
    name = []
    des = []
    num = 0
    with open('data./bd_top3_sample_20180710.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            name.append(dic['name'])
            content = dic['content']
            des_temp = ""
            for item in content:
                des_temp = des_temp + item['title'] + item['desc']
            des.append(des_temp)
            num = num + 1

        for i in range(num):
            corpus.append(des[i])
    return name, des

def main():
    name, corpus = construct_corpus()
    start = datetime.datetime.now()
    vectorizer = TfidfVectorizer(analyzer=pre_process)
    tfidf = vectorizer.fit_transform(corpus)
    print(tfidf.shape)
    words = vectorizer.get_feature_names()
    weights = tfidf.toarray()
    for i in range(len(weights)):
        scores_sklearn = {}
        print(name[i], ':')
        for j in range(len(words)):
            if weights[i][j]:
              scores_sklearn[words[j]] = weights[i][j]

        sorted_words = sorted(scores_sklearn.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:10]:
            print("\tWord: %s, TF-IDF: %.5f" % (word, score))
    end = datetime.datetime.now()
    print ((end-start).total_seconds())

if __name__ == "__main__":
    main()