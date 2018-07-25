import re
import json
import jieba
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer


def get_corpus():

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


def get_stopwords():

    stopwords = []
    path = "./data./stop_words.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def pre_process(text):

    tokens = jieba.lcut(text)
    striped = [word.strip() for word in tokens]
    number_pattern = "^\d*$"
    district_pattern = u'[\u4e00-\u9fa5]{1,7}?(?:省|自治区|市|区|县|镇|村|街|路)$'
    stopwords = get_stopwords()
    filtered = [word for word in striped if len(word) > 1 and not re.match(number_pattern, word)
                and not re.match(district_pattern, word) and word not in stopwords]
    return filtered


def tfidf(corpus):

    vectorizer = TfidfVectorizer(analyzer=pre_process)
    tf_idf = vectorizer.fit_transform(corpus)
    print(tf_idf.shape)
    words = vectorizer.get_feature_names()
    weights = tf_idf.toarray()
    return words, weights


def main():

    names, contents = get_corpus()
    start = datetime.datetime.now()
    words, weights = tfidf(contents)
    path = "./data./output.txt"
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(weights)):
            scores = {}
            print("Top words in", names[i])
            f.write("Top words in " + names[i] + "\n")
            for j in range(len(words)):
                if weights[i][j]:
                    scores[words[j]] = weights[i][j]
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:10]:
                print("\tWord: %s, TF-IDF: %.5f" % (word, score))
                f.write("\tWord: " + word + ", TF-IDF: " + str(round(score, 5)) + "\n")
    end = datetime.datetime.now()
    print ((end-start).total_seconds())

if __name__ == "__main__":
    main()
