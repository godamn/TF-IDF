#import nltk
import math
import string
from collections import Counter
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# def pre_process(text):
#     lowered = text.lower()
#
#     remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
#     removed_punctuation = lowered.translate(remove_punctuation_map)
#
#     tokens = tokenize(removed_punctuation)
#
#     filtered = [word for word in tokens if word not in stopwords.words('english')]
#
#     stemmer = PorterStemmer()
#     stemmed = stem(filtered, stemmer)
#
#     return stemmed


# def tokenize(text):
#     tokens = nltk.word_tokenize(text)
#     return tokens


def stem(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# def count(text):
#     pre_processed = pre_process(text)
#     counted = Counter(pre_processed)
#     return counted


def tf(word, counted):
    return counted[word] / sum(counted.values())


def idf(word, counted_list):
    return math.log(len(counted_list) / (1 + sum(1 for counted in counted_list if word in counted)))


def tfidf(word, counted, counted_list):
    return tf(word, counted) * idf(word, counted_list)


def tfidf_sklearn(texts):
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names()
    weights = tf_idf.toarray()
    return words, weights


def main():
    text1 = "Python is a 2000 made-for-TV horror movie directed by Richard \
    Clabaugh. The film features several cult favorite actors, including William \
    Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy, \
    Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the \
    A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean \
    Whalen. The film concerns a genetically engineered snake, a python, that \
    escapes and unleashes itself on a small town. It includes the classic final\
    girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles, \
     California and Malibu, California. Python was followed by two sequels: Python \
     II (2002) and Boa vs. Python (2004), both also made-for-TV films."

    text2 = "Python, from the Greek word, is a genus of \
    nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are \
    recognised.[2] A member of this genus, P. reticulatus, is among the longest \
    snakes known."

    text3 = "The Colt Python is a .357 Magnum caliber revolver formerly \
    manufactured by Colt's Manufacturing Company of Hartford, Connecticut. \
    It is sometimes referred to as a \"Combat Magnum\".[1] It was first introduced \
    in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued \
    Colt Python targeted the premium revolver market segment. Some firearm \
    collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy \
    Thompson, Renee Smeets and Martin Dougherty have described the Python as the \
    finest production revolver ever made."

    texts = [text1, text2, text3]

    # counted_list = []
    # for text in texts:
    #     counted_list.append(count(text))
    # for i, counted in enumerate(counted_list):
    #     print("Top words in document", i + 1)
    #     scores = {word: tfidf(word, counted, counted_list) for word in counted}
    #     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #     for word, score in sorted_words[:5]:
    #         print("\tWord: %s, TF-IDF: %.5f" % (word, score))

    words, weights = tfidf_sklearn(texts)
    for i in range(len(weights)):
        scores_sklearn = {}
        print("Top words by sklearn in document", i + 1)
        for j in range(len(words)):
            scores_sklearn[words[j]] = weights[i][j]
        sorted_words = sorted(scores_sklearn.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:10]:
            print("\tWord: %s, TF-IDF: %.5f" % (word, score))


if __name__ == "__main__":
    main()
