import nltk
import numpy as np



from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return ps.stem(word.lower())

def list_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(a) for a in tokenized_sentence]

    bag = np.zeros(len(all_words),dtype=np.float32)
    for i , a in enumerate(all_words):
        if a in tokenized_sentence:
            bag[i]=1.0

    return bag

