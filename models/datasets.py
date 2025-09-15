import string
import torch
from torch.utils.data import Dataset
from torch import nn
from nltk.tokenize import word_tokenize
import unicodedata as ud

class QuestionContextDataset(Dataset):
    def __init__(self, questions: list, contexts: list, answerable: list, src_language, tokenizer, translator_model):
        # Translate the questions in English, and tokenize each question into words
        # Vocab will contain the distinct words in the questions
        question_translated = [_translate_to_english(question, src_language, tokenizer, translator_model) for question in questions]
        questions_tokens = [_tokenize_sentence(question) for question in question_translated]
        question_words = [word for word in questions_tokens]
        question_vocab = list(set(question_words))
        self.question_to_idx = {word:i for i,word in enumerate(question_vocab)}
        self.idx_to_question = {i:word for i,word in enumerate(question_vocab)}
        
        # Tokenize each context into words and create vocab of distinct words
        contexts_tokens = [_tokenize_sentence(context) for context in contexts]
        context_words = [word for word in contexts_tokens]
        context_vocab = list(set(context_words))
        self.context_to_idx = {word:i for i,word in enumerate(context_vocab)}
        self.idx_to_context = {i:word for i,word in enumerate(context_vocab)}

        # Inputs of tuples tokenized (question, context) and targets (answerable)
        self.inputs = []
        self.targets = []
        for idx in range(questions_tokens):
            question_input = [self.question_to_idx[word] for word in questions_tokens[idx]]
            context_input = [self.context_to_idx[word] for word in contexts_tokens[idx]]
            answer = answerable[idx]
            
            self.inputs.append(question_input, context_input)
            self.targets.append(answer)
        
        # self.inputs = torch.tensor(self.inputs, dtype=torch.long).view(-1, 1)
        # self.inputs = nn.functional.one_hot(self.inputs, num_classes=len(self.vocab)).float()
        # self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
def _tokenize_sentence(sentence):
    # Remove all punctuation characters, keeping in mind that arabic is written from right to left
    sentence = ''.join([char for char in sentence if not ud.category(char).startswith('P')])
    # Tokenize the question into words
    words = word_tokenize(sentence)
    return words

def _translate_to_english(word, src_lang, tokenizer, model):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(word, return_tensors="pt")
    generated_tokens = model.generate(encoded['input_ids'], forced_bos_token_id=tokenizer.convert_tokens_to_ids('eng_Latn'))
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation