import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktLanguageVars

class TokenizedDocument(object):
    def __init__(self, text):
        self.sentences = []
        self.text = text
        self.preprocessed_split_indices = []
        # this is only optionally kept
        self.sentence_tokens_list = None
        
    def get_sentence_strings(self):
        sentence_strings = []
        for sentence in self.sentences:
            if len(sentence) <= 0:
                continue
            #print(sentence)
            # the 0th element is the start index, so let's get the start of the first and end of the last
            start_idx = sentence[0][0]
            end_idx = sentence[-1][1]
            
            sentence_string = self.text[start_idx : end_idx]
            sentence_strings.append(sentence_string)
        return sentence_strings
        
    def get_longest_sentence_string(self):
        sentence_strings = self.get_sentence_strings()
        longest_sentence_string = ''
        for sentence_string in sentence_strings:
            if len(sentence_string) > len(longest_sentence_string):
                longest_sentence_string = sentence_string
                
        return longest_sentence_string
        
    def get_longest_sentence_token_indices(self):
        sentences_tokens_sorted = self.sentences.copy()
        sentences_tokens_sorted.sort(key = len, reverse = True)
        return sentences_tokens_sorted
        
class ClinicalSentenceTokenizer(object):
    def __init__(self, default_sentence_tokenizer, preprocess_split_re_strs = [r'[\r\n]{2,}']):
        self.punkt_sentence_tokenizer = default_sentence_tokenizer
        
        self.preprocess_split_res = []
        for preprocess_split_re_str in preprocess_split_re_strs:
            compiled_re = re.compile(preprocess_split_re_str, re.MULTILINE)
            self.preprocess_split_res.append(compiled_re)
            
        print('Compiled {0} total preprocessing regular expressions'.format(len(preprocess_split_re_strs)))
        self.last_preprocessed_split_indices = []
        
    def tokenize(self, text):
        # first we go through and used patterns to "preprocess" and find indices where we can break up non-narrative text (non-prose???)
        split_start_idx_set = set()
        for preprocess_split_re in self.preprocess_split_res:
            for match in preprocess_split_re.finditer(text):
                start_idx = match.start()
                split_start_idx_set.add(start_idx)
                
        #print(split_start_idx_set)
        start_index_sorted = sorted(list(split_start_idx_set))
        self.last_preprocessed_split_indices = start_index_sorted
        #print(split_start_idx_set)
        
        if(len(start_index_sorted) == 0):
            # if there was nothing to preprocess, let's allow the entire document to use sentence breaking
            start_index_sorted.append(0)
        
        # now let's break these up and run PUNKT on each one
        all_sentences = []
        for i in range(len(start_index_sorted)):
            # default to the end
            start_idx = start_index_sorted[i]
            end_idx = len(text)
            if i + 1 < len(start_index_sorted):
                end_idx = start_index_sorted[i + 1]
                
            preprocessed_sentence = text[start_idx : end_idx]
            #print('Sentence {0} : [{1}]'.format(i, preprocessed_sentence))
            sentences = self.punkt_sentence_tokenizer.tokenize(preprocessed_sentence)
            all_sentences.extend(sentences)
            
        return all_sentences

class IndexTokenizer(object):
    def __init__(self, sentence_tokenizer, word_tokenizer, keep_token_strings = False):
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer
        self.keep_token_strings = keep_token_strings
        
    def tokenize_document(self, document_text):
        tokenized_doc = TokenizedDocument(document_text)
        
        if self.keep_token_strings:
            tokenized_doc.sentence_tokens_list = []

        # first break into sentences
        sentences = self.sentence_tokenizer.tokenize(document_text)
        
        if hasattr(self.sentence_tokenizer, 'last_preprocessed_split_indices'):
            # store where we made preprocessed splits in case this is helpful
            tokenized_doc.preprocessed_split_indices = self.sentence_tokenizer.last_preprocessed_split_indices
        
        #print(len(sentences))
        
        # keep track of where to start scanning
        scan_idx = 0
        for i in range(len(sentences)):
            sentence = sentences[i]
            # let's try to find this sentence in the text
            #print('Sentence {0} : [{1}]'.format(i, sentence))
            # let's now break this up into words as well
            last_token_end_idx = -1
            
            word_tokens = None
            if hasattr(self.word_tokenizer, 'word_tokenize'):
                word_tokens = self.word_tokenizer.word_tokenize(sentence)
            elif hasattr(self.word_tokenizer, 'tokenize'):
                word_tokens = self.word_tokenizer.tokenize(sentence)
            else:
                raise ValueError('Cannot tokenize with an object that does not define word_tokenize() ot tokenize()')
            sent_word_indices = []
            for word_token in word_tokens:
                # if we walk through left-to-right we can re-assemble the indices for these tokens
                token_start_idx = document_text.find(word_token, scan_idx)
                token_end_idx = token_start_idx + len(word_token)
                # update where we scan from (end of the current word)
                scan_idx = token_end_idx
                
                # give back indices which are relative to the document rather than within-sentence
                token_offsets = (token_start_idx, token_end_idx)
                sent_word_indices.append(token_offsets)
                
            tokenized_doc.sentences.append(sent_word_indices)
            if self.keep_token_strings:
                tokenized_doc.sentence_tokens_list.append(word_tokens)
            
        return tokenized_doc
        
    def tokenize_documents(self, document_texts):
        for document_text in document_texts:
            yield self.tokenize_document(document_text)