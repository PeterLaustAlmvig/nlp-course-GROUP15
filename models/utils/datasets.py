import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
import unicodedata as ud

class WordDataset(Dataset):
    """Sentence Dataset for training models on predicting the likelihood of an entire piece of text,
    the pieces are tokenized using the English word_tokenize meaning that they are split mainly using spaces.
    
    It is possible to specify a pre-generated vocab in case the model has been trained to only allow certain words,
    otherwise if vocab is None then it will generate the vocab based on the supplied sentences in sentence_data.
    
    The sentences are encoded based on each words presence in the word_to_idx/vocab, and words that are not present will
    be set to -1, the same system is applied when creating the target tensor giving each word a score of either 1 or -1.
    """
    def __init__(self, sentence_data: list, vocab=None, word_to_idx=None, idx_to_word=None):
        # Tokenize each sentence and generate the vocab if not given
        sentences_tokens = [_tokenize_sentence(question) for question in sentence_data]
        if vocab == None:
            self.vocab, self.word_to_idx, self.idx_to_word = _generate_vocab_keying(sentences_tokens)
        else:
            self.vocab = vocab
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
        
        # List of lists for the tokenized words
        self.sentences = sentences_tokens

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_sentence = torch.LongTensor([
            self.words_to_idx[word] if word in self.word_to_idx.keys() else -1
            for word in sentence])
        target_sentence = torch.LongTensor([
            1 if word in self.word_to_idx.keys() else -1 
            for word in sentence])
        return input_sentence, target_sentence
    
def _tokenize_sentence(sentence):
    # Remove all punctuation characters, keeping in mind that arabic is written from right to left
    sentence = ''.join([char for char in sentence if not ud.category(char).startswith('P')])
    # Tokenize the question into words
    words = word_tokenize(sentence)
    return words

def _generate_vocab_keying(sentences_tokens):
    words = [word for word in sentences_tokens]
    vocab = list(set(words))
    word_to_idx = {word:i for i,word in enumerate(vocab)}
    idx_to_word = {i:word for i,word in enumerate(vocab)}
    return vocab, word_to_idx, idx_to_word