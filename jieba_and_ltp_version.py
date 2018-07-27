import jieba
from pyltp import Postagger
import os

MODELDIR = "ltp_data"

def fenci_ltp():
    fin = open('input.txt', 'r')    # 需要进行分词的文件，每行一句话

    jieba.load_userdict('mydict.txt')
    postagger = Postagger()    # 初始化实例
    postagger.load(os.path.join(MODELDIR, "pos.model"))    # 加载模型

    for eachLine in fin:
        line = eachLine.strip()
        words = jieba.cut(line)    # jieba分词返回的是可迭代的generator，里面的词是unicode编码
        words = [word.encode('utf-8') for word in words]    # 将unicode编码的单词以utf-8编码
        postags = postagger.postag(words)    # 词性标注
        words_postags = []
        for word,postag in zip(words, postags):
            words_postags.append(word + '/' + postag)
        print(' '.join(words_postags))

    postagger.release()    # 释放模型

if __name__ == '__main__':
    fenci_ltp()