import nltk

# now let's set up some new feature extraction methods which can leverage word embeddings
class CRFFeatureExtractor(object):

    def __init__(self, word_vector_map = None,
                 window_size = 2,
                 char_ngrams_enabled = set([2, 3]),
                 enable_engineered_features = True,
                enable_pos_features = False,
                 pos_tagger = None,
                 enable_stem_features = False,
                max_upper_case_ratio = 0.6):
        self.word_vector_map = word_vector_map
        self.enable_engineered_features = enable_engineered_features
        self.window_size = window_size
        self.char_ngrams_enabled = char_ngrams_enabled
        self.enable_pos_features = enable_pos_features
        self.pos_tagger = pos_tagger
        self.enable_stem_features = enable_stem_features
        self.max_upper_case_ratio = max_upper_case_ratio
        # not the most accurate stemmer, but we want this to be fast
        self.stemmer = nltk.stem.porter.PorterStemmer()
        
        if self.word_vector_map is not None:
            if hasattr(word_vector_map, 'vector_size'):
                self.vector_size = word_vector_map.vector_size
            else:
                first_vector = list(word_vector_map.values())[0]
                self.vector_size = first_vector.shape[0]
            print('Ready to extract for embeddings of size : {0}'.format(self.vector_size))
        
    def add_embedding_features(self, prefix, word, features):
        if self.word_vector_map is None:
            return
        
        word_key = word
        if word_key not in self.word_vector_map:
            # back off to lowercase form
            word_key = word.lower()
            
        if word_key not in self.word_vector_map:
            return
        
        word_vector = self.word_vector_map[word_key]
        
        for i in range(self.vector_size):
            dim_key = '{0}{1}'.format(prefix, i)
            features[dim_key] = word_vector[i]

    def word2features(self, sent, i):
        
        # let's work across the window of tokens specified
        features = {}
        
        # let's see if there are any senteclevel (not window level) features we want to useCE_
        sentence_tokens_lower = [x[0].lower() for x in sent]
        sentence_string_lower = ' '.join(sentence_tokens_lower)
        
        # let's count the uppercase tokens
        sentence_tokens_original_case = [x[0] for x in sent]
        uppercase_list = [1 for x in sentence_tokens_original_case if x.isupper() ]
        
        upper_ratio = float((len(uppercase_list))) / float(len(sentence_tokens_original_case))
        use_lower_tokens = upper_ratio > self.max_upper_case_ratio
        
        sentence_tokens_preferred = sentence_tokens_original_case
        if use_lower_tokens:
           sentence_tokens_preferred = sentence_tokens_lower
        
        # make sure that we have no 0-length tokens, since it breaks hte 
        if self.enable_pos_features and self.enable_engineered_features:
            for token_idx in range(len(sentence_tokens_preferred)):
                token = sentence_tokens_preferred[token_idx]
                if len(token) <= 0:
                    # put this back so that the tagger will not crash
                    sentence_tokens_preferred[token_idx] = ' '
            
        sentence_pos_tag_pairs = None
        if self.enable_pos_features and self.enable_engineered_features:
            sentence_pos_tag_pairs = self.pos_tagger.tag(sentence_tokens_preferred)
        
        # let's get tokens on either side
        tokens_left = sentence_tokens_lower[:i]
        tokens_right = tokens_left = sentence_tokens_lower[i + 1:]
        
        if '?' in tokens_left:
            features['?_left'] = 1.0
        if '?' in tokens_right:
            features['?_right'] = 1.0
        
        for window_idx in range(-1 * self.window_size, self.window_size + 1):
            curr_token_idx = i + window_idx
            #print('Working on token index {0}'.format(curr_token_idx))
            prefix = str(window_idx)

            if curr_token_idx < 0:
                if self.enable_engineered_features:
                    features[prefix + 'word.lower()'] = 'bos'      
            elif curr_token_idx >= len(sent):
                if self.enable_engineered_features:
                    features[prefix + 'word.lower()'] = 'eos' 
            else:
                word = sentence_tokens_preferred[curr_token_idx]

                self.add_embedding_features('{0}E'.format(prefix), word, features)
                
                word_lower = word.lower()
                if self.enable_engineered_features:
                    features[prefix + 'word.lower()'] = word_lower
                    if 2 in self.char_ngrams_enabled:
                        features[prefix + 'word[-2:]'] = word_lower[-2:]
                        features[prefix + 'word[:2]'] = word_lower[:2]
                    if 3 in self.char_ngrams_enabled:
                        features[prefix + 'word[-3:]'] = word_lower[-3:]
                        features[prefix + 'word[:3]'] = word_lower[:3]
                    if 4 in self.char_ngrams_enabled:
                        features[prefix + 'word[-4:]'] = word_lower[-4:]
                        features[prefix + 'word[:4]'] = word_lower[:4]
                    
                    features[prefix + 'word.islower()'] = word.islower()
                    features[prefix + 'word.isupper()'] = word.isupper()
                    features[prefix + 'word.istitle()'] = word.istitle()
                    features[prefix + 'word.isdigit()'] = word.isdigit()
                    features[prefix + 'word_any_alpha'] = any(char.isalpha() for char in word_lower) 
                    features[prefix + 'word_all_punct'] = all(char in string.punctuation for char in word_lower)
                    if self.enable_stem_features:
                        features[prefix + 'wordstem'] = self.stemmer.stem(word_lower)
                
                if self.enable_pos_features and self.enable_engineered_features:
                    postag_pair = sentence_pos_tag_pairs[curr_token_idx]
                    # This pair will be in the form ('token', 'TAG')
                    postag =  postag_pair[1]
                    features[prefix + 'postag'] = postag  
            # update for any word encountered
            #features.update(word_features)

        return features
    
    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]