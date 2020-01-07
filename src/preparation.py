
PAD_INDEX = 0
UNK_INDEX = 1

class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX, "UNK":UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX:"PAD", UNK_INDEX:"UNK"}
        self.n_words = 2
        self.word_num = 0
    def index_words(self, sentence):
        for word in sentence:
            self.word_num+=1
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1
