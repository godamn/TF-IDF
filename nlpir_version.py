import re
import json
import pynlpir
import datetime
import sys
import threading

from sklearn.feature_extraction.text import TfidfVectorizer

#V1版本去除人名，地名等常见词性
#V2版本增加大量黑名单词性
#V3去除特殊符号
#V4增加大写转换小写，增加公司名称
#V5在百度前三页文档中去除公司名的影响

#读取数据构建语料库
def get_corpus():
    print("get corpos...")
    names = []
    contents = []
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
                #desc = desc + dict['name']
                desc.replace(dict['name'], "")
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

#构建TfidfVectorizer模型
def tfidf(corpus):

    vectorizer = TfidfVectorizer(analyzer=pre_process)
    tf_idf = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    with open('./datasample./10000_key_world_list_nlpirV5.txt', 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word)
            f.write('\n')
    weights = tf_idf.tolil()#稀疏矩阵转换格式加速读取
    return words, weights



def main():

    path = "./output./10000_result_with_pynlpirV5.txt"
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
            print("Top words in", names[i])
            f.write("Top words in " + names[i] + "\n")
            for j in range(len(words)):
                temp = weights[i,j]
                if temp:
                    scores[words[j]] = temp
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:10]:
                print("\tWord: %s, TF-IDF: %.5f" % (word, score))
                f.write("\tWord: " + word + ", TF-IDF: " + str(round(score, 5)) + "\n")
        print("finish writing...")
        end = datetime.datetime.now()
        print("Writing Time: %.5fs" % ((end - start).total_seconds()))
        f.write("Writing Time: %.5fs" % ((end - start).total_seconds()) + '\n')
        pynlpir.close()

if __name__ == "__main__":
    main()
