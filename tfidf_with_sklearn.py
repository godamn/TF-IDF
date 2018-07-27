import re
import json
import jieba
import datetime
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer

#读取数据构建语料库
def get_corpus():
    print("get corpos...")
    names = []
    contents = []
    path = "./data./bd_top3_sample_20180710.json"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            dict = json.loads(line)
            names.append(dict['name'])
            desc = ""
            for item in dict['content']:
                desc = desc + item['desc'] + "\n"
            contents.append(desc)
    return names, contents

#加载停用词表
def get_stopwords():

    stopwords = []
    path = "./data./stop_words.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

stopwords = get_stopwords()

#切词与过滤词预处理
def pre_process(text):

    tokens = jieba.lcut(text, HMM=True)
    striped = [word.strip() for word in tokens]
    number_pattern = "^\d*$"#过滤数字串
    district_pattern = u'[\u4e00-\u9fa5]{1,7}?(?:省|自治区|市|区|县|镇|村|街|路)$'#过滤地区
    filtered = [word for word in striped if len(word) > 1 and not re.match(number_pattern, word)
                and not re.match(district_pattern, word) and word not in stopwords]
    return filtered

#构建TfidfVectorizer模型
def tfidf(corpus):

    vectorizer = TfidfVectorizer(analyzer=pre_process)
    tf_idf = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    weights = tf_idf.tolil()#稀疏矩阵转换格式加速读取
    return words, weights


def main():
    path = "./output./100_result_with_sklearn.txt"
    with open(path, 'w', encoding='utf-8') as f:
        print("Getting Corpus...")
        start = datetime.datetime.now()
        names, contents = get_corpus()
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

        print("Writing Data...")
        start = datetime.datetime.now()

        for i in range(len(names)):
            scores = {}
            #print("Top words in", names[i])
            f.write("Top words in " + names[i] + "\n")
            for j in range(len(words)):
                if weights[i,j]:
                    scores[words[j]] = weights[i,j]
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:10]:
                #print("\tWord: %s, TF-IDF: %.5f" % (word, score))
                f.write("\tWord: " + word + ", TF-IDF: " + str(round(score, 5)) + "\n")

        print("finish writing...")
        end = datetime.datetime.now()
        print("Writing Time: %.5fs" % ((end - start).total_seconds()))
        f.write("Writing Time: %.5fs" % ((end - start).total_seconds()) + '\n')

if __name__ == "__main__":
    main()
