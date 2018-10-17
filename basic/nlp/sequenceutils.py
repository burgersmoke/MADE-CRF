def get_sentence_bio_tagged_tokens(annotated_doc):
    # an AnnotatedDocument contains original text, tokenized indices and annotations so let's return 2 lists:
    # lists of sentence tokens.  Ex : [['I', 'am', 'a', 'sentence'], ['I', 'am', 'too']]
    # lists of entity type tags . Ex : [['O', 'O', 'O', 'SOMETHING'], ['O', 'O', 'O']]
    
    token_lists = []
    tag_lists = []
    # loop by sentence
    for sentence in annotated_doc.tokenized_doc.sentences:
        #print(sentence)
        sentence_tokens = []
        sentence_tags = []
        for token_offset_pair in sentence:
            token_start_idx = token_offset_pair[0]
            token_end_idx = token_offset_pair[1]
            token_str = annotated_doc.text[token_start_idx : token_end_idx]
            sentence_tokens.append(token_str)
            
            tag = 'O'
            # for each token, let's see what entity type label should be applied
            for annotation in annotated_doc.annotations:
                # is this token completely subsumed by this larger annotation?
                if annotation.start_index < token_start_idx and annotation.end_index > token_end_idx:
                    tag = annotation.type
                    break
                # or does it line up exactly with one of the token's boundaries?
                elif annotation.start_index == token_start_idx or annotation.end_index == token_end_idx:
                    tag = annotation.type
                    break
                # otherwise, if it intersects somewhere in between, let's call it this annotation
                elif annotation.start_index > token_start_idx and annotation.end_index < token_end_idx:
                    tag = annotation.type
                    break
                    
            sentence_tags.append(tag)
            
        # before we finish up with these tags, let's convert them into BIO format (Beginning, Inside, Outside)
        sentence_bio_tags = []
        for i in range(len(sentence_tags)): 
            curr_tag = sentence_tags[i]
            prev_tag = 'O'
            
            if i > 0:
                prev_tag = sentence_tags[i - 1]
                
            if curr_tag != 'O':
                if curr_tag == prev_tag:
                    curr_tag = 'I-' + curr_tag
                else:
                    curr_tag = 'B-' + curr_tag
                    
            sentence_bio_tags.append(curr_tag)
            
        # add all sentence data to each
        token_lists.append(sentence_tokens)
        tag_lists.append(sentence_bio_tags)
            
    return token_lists, tag_lists