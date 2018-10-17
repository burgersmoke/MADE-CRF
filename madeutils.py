import os
import json
import numpy as np
import itertools
import collections
import glob

import bioc

# evaluation from the challenge itself
from bioc_evaluation import get_f_scores

import basic
from basic.nlp.tokenizers import clinical_tokenizers
from basic.nlp.annotation.annotation import Annotation, AnnotatedDocument
from basic.nlp.sequenceutils import get_sentence_bio_tagged_tokens

# this package does sequence labels with CRF so we will use it for metrics 
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.utils import flatten

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# now let's load in all documents (both text and annotations)
def read_made_data(made_base_dir, index_tokenizer, ignore_labels = ['PHI'], verbose = False):

    annotated_docs = []
    corpus_path = made_base_dir + r'\corpus'
    for f in os.listdir(corpus_path):
        corpus_file_path = os.path.join(corpus_path, f)
        if not os.path.isfile(corpus_file_path):
            continue

        corpus_file = open(corpus_file_path, 'r', encoding='utf-8')

        text = corpus_file.read()
        
        corpus_file.close()
        
        annotated_doc = AnnotatedDocument()
        annotated_doc.text = text
        annotated_doc.filename = f
        # now tokenize by sentence and tokens and store this
        annotated_doc.tokenized_doc = index_tokenizer.tokenize_document(text)
        
        #print('Document total Sentences : {}'.format(len(annotated_doc.tokenized_doc.sentences)))
        
        # we also need to read in the BIOC annotations...
        bioc_file_name = os.path.join(os.path.join(made_base_dir, 'annotations'), f + '.bioc.xml')
        
        # add this even if we don't end up finding a BIOC file
        annotated_docs.append(annotated_doc)
        
        if not os.path.isfile(bioc_file_name):
            print('No BIOC annotations for file : {}'.format(bioc_file_name))
            continue
        
        if verbose:
            print('Loading annotations for {}'.format(bioc_file_name))
            
        with bioc.iterparse(bioc_file_name) as parser:
            collection_info = parser.get_collection_info()
            for document in parser:
                i = 1
                #print(document)

                for passage in document.passages:
                    #print(len(passage.annotations))
                    for bioc_annotation in passage.annotations:
                        locations = bioc_annotation.locations
                        if len(locations) != 1:
                            print('Expected Annotation to have 1 Locations, but got {}'.format(len(locations)))
                            continue
                        # just use the first since we expect only 1
                        location = locations[0]
                        anno = Annotation()
                        anno.start_index = location.offset
                        anno.end_index = location.offset + location.length
                        anno.spanned_text = bioc_annotation.text

                        #print(bioc_infons)
                        if 'type' in bioc_annotation.infons:
                            anno.type = bioc_annotation.infons['type']
                            # let's see if we should ignore this type
                            for ignore_label in ignore_labels:
                                if anno.type == ignore_label:
                                    anno.type = 'O'
                                    break
                        else:
                            anno.type = 'UNK'
                        
                        #print(anno)
                        #break
                        
                        annotated_doc.annotations.append(anno)
                        
    print('Total annotated documents loaded : {}'.format(len(annotated_docs)))
        
    return annotated_docs
    
def count_labels(labels):
    labels_flat = flatten(labels)
    counter = collections.Counter(labels_flat)
    print(counter)
    
def get_all_sentence_tokens_and_tags(annotated_docs):
    X = []
    y = []
    for annotated_doc in annotated_docs:
        doc_tokens, doc_tags = get_sentence_bio_tagged_tokens(annotated_doc)
        for i in range(len(doc_tokens)):
            sentence_tokens = doc_tokens[i]
            sentence_tags = doc_tags[i]
            if len(sentence_tokens) > 0 and len(sentence_tags) > 0:
                X.append(sentence_tokens)
                y.append(sentence_tags)
                
    return X, y
    
# NOTE : This plotting function below came from here:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize = (8,8)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def get_coarse_labels(original_labels):
    coarse_labels = []
    for original_label in original_labels:
        coarse_label = original_label.replace('B-', '').replace('I-', '')
        coarse_labels.append(coarse_label)
    return coarse_labels

def gather_validation_metrics(X_text, y, tokenizer, model, preprocessor, 
    batch_size = 128, check_lengths = False, verbose = False, fine_label_report = False, dataset = 'UNKNOWN DATASET',
    label_ignore_set = set(['O'])):
    gold_labels = []
    pred_labels = []
    gold_coarse_labels = []
    pred_coarse_labels = []
    
    # now put all the predictions together
    fine_labels_set = set()
    coarse_labels_set = set()
    MAX_PREDICTIONS = len(X_text)
    # do all predictions at once (rather than sentence-by-sentence) and then collect the results
    sentence_preds = model.predict(preprocessor.transform(X_text[:MAX_PREDICTIONS]),
                                                batch_size = batch_size
                                               )
        
    # then go back and get predictions one at a time
    for i in range(len(X_text[:MAX_PREDICTIONS])):
        # only one sentence, so grab the 0 row...
        sentence_length = len(y[i])
        sentence_pred = sentence_preds[i, :sentence_length]
        #print(sentence_pred)
        #break
        
        # prep the labels
        sentence_coarse_labels = get_coarse_labels(y[i])

        # get the predictions
        pred_sentence_labels = preprocessor.inverse_transform(np.argmax(sentence_pred, -1))
        
        pred_sentence_coarse_labels = get_coarse_labels(pred_sentence_labels)
        
        if check_lengths:
            if len(y[i]) == len(pred_sentence_labels):
                print('MATCH of GOLD and PRED')
            else:
                print('NO MATCH ON GOLD and PRED')
                print('GOLD : {0}, PRED : {1}'.format(len(y[i]), len(pred_sentence_labels)))
                print('SENTENCE PRED LENGTH : {}'.format(len(sentence_pred)))
                print(X_text[i])
                print(y[i])
                print(pred_sentence_labels)
        
        # the sklearn-crfsuite metrics call its flatten() to convert a list of lists to one flat list 
        gold_labels.append(y[i])
        gold_coarse_labels.append(sentence_coarse_labels)
        pred_labels.append(pred_sentence_labels)
        pred_coarse_labels.append(pred_sentence_coarse_labels)
        
        fine_labels_set |= set(y[i])
        coarse_labels_set |= set(sentence_coarse_labels)
        
        #print(pred_sentence_labels)
        #print(y[i])
        
    print('Reporting metrics for dataset : [{0}]'.format(dataset))
    if verbose:
        print('Total gold FINE : {}'.format(len(gold_labels)))
        print('Total pred FINE : {}'.format(len(pred_labels)))
        print('Total gold COARSE : {}'.format(len(gold_coarse_labels)))
        print('Total pred COARSE : {}'.format(len(pred_coarse_labels)))

        print('Total FLATTENED gold FINE : {}'.format(len(flatten(gold_labels))))
        print('Total FLATTENED pred FINE : {}'.format(len(flatten(pred_labels))))
        print('Total FLATTENED gold COARSE : {}'.format(len(flatten(gold_coarse_labels))))
        print('Total FLATTENED pred COARSE : {}'.format(len(flatten(pred_coarse_labels))))
        
    # now let's do some evaluation
    fine_labels_sorted_list = sorted(list(fine_labels_set))
    coarse_labels_sorted_list = sorted(list(coarse_labels_set))
    
    # pull out any labels that we want to ignore
    fine_labels_sorted_list = [x for x in fine_labels_sorted_list if x not in label_ignore_set]
    coarse_labels_sorted_list = [x for x in coarse_labels_sorted_list if x not in label_ignore_set]
    
    if fine_label_report:
        print('FINE label report : ')
        print(flat_classification_report(gold_labels, pred_labels, labels = fine_labels_sorted_list, digits=3))
        
    print('COARSE label report : ')
    print(flat_classification_report(gold_coarse_labels, pred_coarse_labels, labels = coarse_labels_sorted_list, digits=3))
    
    flat_gold_coarse_labels = flatten(gold_coarse_labels)
    flat_pred_coarse_labels = flatten(pred_coarse_labels)
    
    coarse_labels_sorted_list_nofilter = sorted(list(coarse_labels_set))
    confusion_coarse = confusion_matrix(flat_gold_coarse_labels, flat_pred_coarse_labels, labels = coarse_labels_sorted_list_nofilter)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_coarse, classes=coarse_labels_sorted_list_nofilter,
                          title='Coarse Label Confusion for : [{0}]'.format(dataset))
    plt.show()
    
def evaluate_via_bioc(test_docs, crf, extractor, prediction_dir, made_base_dir = None):
    print('Total documents for evaluation : {}'.format(len(test_docs)))
    
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
        
    existing_files = glob.glob('{0}/*'.format(prediction_dir))
    existing_files_removed = 0
    for f in existing_files:
        os.remove(f)
        existing_files_removed += 1
        
    print('Existing files removed : {}'.format(existing_files_removed))
    
    prediction_documents_written = 0
    reference_filenames = []
    for test_doc in test_docs:
        #print('Working on document : {}'.format(test_doc.filename))
        
        collection = bioc.BioCCollection()
        document = bioc.BioCDocument()
        document.id = test_doc.filename
        collection.add_document(document)
        passage = bioc.BioCPassage()
        passage.offset = 0
        document.add_passage(passage)
        
        next_annotation_id = 1
        
        # now an annotation can be written for each label prediction
        for sentence in test_doc.tokenized_doc.sentences:
            sentence_tokens = []
            # gather tokens in a sentence
            for token_offset_pair in sentence:
                token = test_doc.text[token_offset_pair[0] : token_offset_pair[1]]
                sentence_tokens.append(token)
            if len(sentence_tokens) == 0:
                continue
                
            sentence_features = extractor.sent2features(sentence_tokens)
            sentence_pred = crf.predict([sentence_features])[0]
            
            if len(sentence_pred) != len(sentence):
                print('Sentence Features Length : {}'.format(len(sentence_features)))
                print('Sentence Pred Length : {}'.format(len(sentence_pred)))
                print('Sentence Length : {}'.format(len(sentence)))
            
            # walk manually through the predictions and add spans as appropriate
            token_idx = 0
            while token_idx < len(sentence_pred):
                token_pred = sentence_pred[token_idx]
                if token_pred != 'O':
                    base_label = token_pred.replace('B-', '').replace('I-', '')
                    start_offset = sentence[token_idx][0]
                    end_offset = sentence[token_idx][1]
                    # now let's look to the right as long as we see tokens which are part of this same label
                    while token_idx + 1 < len(sentence_pred) and sentence_pred[token_idx + 1] == ('I-' + base_label):
                        # advance the token
                        token_idx += 1
                        # update the end of this span
                        end_offset = sentence[token_idx][1]
                        
                    # finally we have an annotation that we can add
                    annotation = bioc.BioCAnnotation()
                    
                    annotation.infons['type'] = base_label
                    annotation.text = test_doc.text[start_offset : end_offset]
                    # current reference replaces newlines with literal '\n'
                    annotation.text = annotation.text.replace('\n', '\\n').replace('\r', '\\r')
                    annotation.id = str(next_annotation_id)
                    location = bioc.BioCLocation(start_offset, end_offset - start_offset)
                    
                    next_annotation_id += 1
                    annotation.add_location(location)
                    passage.add_annotation(annotation)
                    
                # advance the token no matter what happened above
                token_idx += 1
        
        prediction_filename = os.path.join(prediction_dir, '{}.bioc.xml'.format(test_doc.filename))
        
        if made_base_dir is not None:
            reference_filename = os.path.join(os.path.join(made_base_dir, 'annotations'), '{}.bioc.xml'.format(test_doc.filename))
            reference_filenames.append(reference_filename)
        
        with open(prediction_filename, 'w') as fp:
            bioc.dump(collection, fp)
            prediction_documents_written += 1
            
    print('Total prediction documents written : {}'.format(prediction_documents_written))
    
    # finally we can invoke some evaluation (if enabled)
    if made_base_dir is not None:
        annotation_dir = os.path.join(made_base_dir, 'annotations')
        text_dir = os.path.join(made_base_dir, 'corpus')
        # first param can be an actual directory (string) or a list of filepaths
        get_f_scores(reference_filenames, prediction_dir, text_dir)