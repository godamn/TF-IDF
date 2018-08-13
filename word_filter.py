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
                desc = desc.replace("[", "").replace("]", "").replace("\n", "。")
                contents.append(desc)
    return names, contents

#加载停用词表
def get_stopwords_and_types():

    stopwords = []
    stoptypes = []
    path = "./data./stop_words_thulac.txt"
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
    for types in stoptypes:
        print(types)
    return stopwords, stoptypes

stop_words, stop_types = get_stopwords_and_types()
pynlpir.open()

word2type = {}
#切词与过滤词预处理
def pre_process(text):
    useful_word_pattern = u"^[a-zA-Z0-9\u4e00-\u9fa5]+$"
    district_pattern = u"[\u4e00-\u9fa5]{1,7}?(?:省|自治区|市|区|县|镇|村|街)$"
    tokens = pynlpir.segment(text, pos_names='child')
    filtered = [word[0] for word in tokens
                if len(word[0]) > 1 and len(word[0]) < 9 and not re.match(district_pattern, word[0]) and
                word[1] not in stop_types and word[0] not in stop_words and re.match(useful_word_pattern, word[0])]

    return filtered

#构建TfidfVectorizer模型
def tfidf(corpus):
    word_type = []
    vectorizer = TfidfVectorizer(analyzer=pre_process)
    tf_idf = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    weights = tf_idf.tolil()#稀疏矩阵转换格式加速读取
    with open('./datasample./1000_wordtype_list_without_n_vV3.txt', 'w', encoding='utf-8') as word_file:
        final_word2type = {}
        for word in words:
            temp = word2type[word]
            final_word2type[word] = temp
            if temp not in word_type:
                word_type.append(temp)

        for t in word_type:
            print(t)
            for word in words:
                if final_word2type[word] == t:
                    word_file.write(word)
                    word_file.write(" : ")
                    word_file.write(str(final_word2type[word]))
                    word_file.write("\n")

    return words, weights


def main():
    path = "./output./1000_result_with_pynlpir.txt"
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
        pynlpir.close()

if __name__ == "__main__":
    main()
