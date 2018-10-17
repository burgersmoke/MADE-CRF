############## MADE 1.0 Evaluation Script ####################
#
#
# This is an evaluation script for the three tracks of MADE 1.0 challange held 
# by the BioNLP Lab at UMass. 
#
# Usage: 
# For concise output 
# > python bioc_evaluation.py annotation_dir prediction_dir text_dir
# For verbose output 
# > python bioc_evaluation.py annotation_dir prediction_dir text_dir v
#
# Dependency for this script :
# BioC package from https://github.com/yfpeng
# Installation instructions at https://github.com/yfpeng/bioc
#
#
# It provides output scores in three modes.
# 1) Strict NER : NER evaluation at phrase level. A prediction is classified 
# as correct only if the phrase span and type are correct. The phrase
# span is evaluated as correct even if there is a +/-1 error in end boundary.
# This relaxation is provided to ensure that inconsistent trailing punctuation
# in some annotations does not negatively effect the score. 
#
# 2) Relaxed NER : NER evaluation at word level. This metric is not considered
# for evaluation in MADE 1.0, it provided for self evaluation. A very simple 
# tokenizer is used to split the phrase into words.
#
# 3) Strict Relation: Relation evaluation. A relation is classified as 
# correct only if both NER annotations are correct according to Strict NER and
# the relation type is also correct.
#
#
#
# ONLY THE FOLLOWING LABELS ARE EVALUATED. ALL OTHERS WILL BE IGNORED

RELATION_TYPE=['severity_type','manner/route','reason','do','du','fr','adverse']  # Legal Relation labels
ANNOTATION_TYPE=['Drug','Indication','Frequency','Severity','Dose','Duration','Route','ADE','SSLIF'] # Legal NER annotation labels

try:
    import bioc 
except ImportError:
    raise ImportError('The python package Bioc is a dependency and is not installed. See https://github.com/yfpeng/bioc for installation instructions') 
import os
import sys
import logging

if len(sys.argv)==5 and sys.argv[4]=='v':
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

def generate_relaxed_annotation_variants(annotation_type,offset,length):
    # producing relaxed variants for matching with annotations containing trailing punctuation or spaces.
    return [(annotation_type,offset,length+i) for i in [-1,0,1]] 


def generate_relaxed_relation_variants(relation_type,annotation_list1,annotation_list2):
    # producing relaxed variants for matching with annotations containing trailing punctuation or spaces.
    relation_variants=[]
    for entry1 in annotation_list1:
        for entry2 in annotation_list2:
            relation_variants.append((relation_type,entry1,entry2))
    return relation_variants


def read_prediction(infile):
    with open(infile,'r') as fin:
        try:
            collection=bioc.load(fin)
        except:
            logging.error('BioC file {0} not well formed'.format(infile))
            raise
    assert collection.documents.__len__() !=0, "Each document should be encoded in its own collection"
    annotations,relations={},{}
    for passage in collection.documents[0].passages:
        for annotation in passage.annotations:
            assert annotation.id not in annotations,'Duplicate annotation id found. Please verify{0}'.format(annotation.id)
            if annotation.infons['type'] in ANNOTATION_TYPE:
                annotations[annotation.id]=(annotation.infons['type'],annotation.locations[0].offset,annotation.locations[0].length)

    for passage in collection.documents[0].passages:
        for relation in passage.relations:
            assert relation.id not in relations,'Duplicate relation id found. Please verify{0}'.format(relation.id)
            if relation.infons['type'] in RELATION_TYPE:
                if relation.nodes[0].refid in annotations and relation.nodes[1].refid in annotations: 
                    # Disregarding relations that have illegal annotation ids.
                    annotation1=annotations[relation.nodes[0].refid] 
                    annotation2=annotations[relation.nodes[1].refid]
                    relations[relation.id]=(relation.infons['type'],annotation1,annotation2)
                else:
                    logging.debug('Disregarding relation id {0} from file {1} because annotation entries are not valid'.format(relation.id,infile))

    return annotations,relations

def read_reference(infile):
    with open(infile,'r') as fin:
        try:
            collection=bioc.load(fin)
        except:
            logging.error('BioC file {0} not well formed'.format(infile))
            raise
    assert collection.documents.__len__() !=0, "Each document should be encoded in its own collection"
    annotations,relations={},{}
    for passage in collection.documents[0].passages:
        for annotation in passage.annotations:
            if annotation.infons['type'] in ANNOTATION_TYPE:
                annotations[annotation.id]=generate_relaxed_annotation_variants(annotation.infons['type'],annotation.locations[0].offset,annotation.locations[0].length)
    for passage in collection.documents[0].passages:
        for relation in passage.relations:
            if relation.infons['type'] in RELATION_TYPE:
                relations[relation.id]=generate_relaxed_relation_variants(relation.infons['type'],annotations[relation.nodes[0].refid],annotations[relation.nodes[1].refid])

    annotation_map={ann : key for (key,ann_list) in annotations.items() for ann in ann_list}
    relation_map={rel : key for (key,rel_list) in relations.items() for rel in rel_list}

    
    return annotations,relations,annotation_map,relation_map


def split_annotations(annotation_id,category,offset,annotation_length,text):
    # Split NER annotations on spaces for approximate/word level match
    words=[]
    current_index=-1
    length=0
    for idx in range(offset,offset+annotation_length):
        if current_index == -1:
            if text[idx]!=' ':
                current_index=idx
                length+=1
        elif current_index!=-1:
            if text[idx]==' ':
                words.append((current_index,length))
                current_index=-1
                length=0
            else:
                length+=1
    if current_index!=-1:
        words.append((current_index,length))
    return_packet=[(annotation_id+'-'+str(idx),(category,word[0],word[1])) for idx,word in enumerate(words)]
    return return_packet
        
    

def read_word_based_annotations(infile,textfile,prediction_file=False):
    with open(infile,'r') as fin:
        try:
            collection=bioc.load(fin)
        except:
            logging.error('BioC file {0} not well formed'.format(infile))
            raise
    with open(textfile,'r') as fin:
        text=fin.read()
    assert collection.documents.__len__() !=0, "Each document should be encoded in its own collection"

    annotations,seen_anns={},set()
    for passage in collection.documents[0].passages:
        for annotation in passage.annotations:
            assert annotation.id not in seen_anns,'Duplicate annotation id found. Please verify{0}'.format(annotation.id)
            if annotation.infons['type'] in ANNOTATION_TYPE:
                word_annotations=split_annotations(annotation.id,annotation.infons['type'],annotation.locations[0].offset,annotation.locations[0].length,text)
                for word_id,word_annotation in word_annotations:
                    annotations[word_id]=word_annotation

    annotation_map={ann : key for key,ann in annotations.items()}
    return annotations,annotation_map


def generate_match(categories,reference_map,predictions):
    reference_set={}
    for entry,entry_id in reference_map.items():
        if reference_map[entry] not in reference_set:
            reference_set[entry_id]=entry[0]
 
    true_positives={categ:0 for categ in categories}
    false_positives={categ:0 for categ in categories}
    false_negatives={categ:0 for categ in categories}

    for pred_id,prediction in predictions.items():
        assert prediction[0] in categories,"Illegal category {0}. Please verify {1}".format(prediction[0],pred_id)

        if prediction in reference_map:
            #This is a true positive
            true_positives[prediction[0]]+=1
            if reference_map[prediction] in reference_set:
                del reference_set[reference_map[prediction]]
        else:
            #This is a false positive
            false_positives[prediction[0]]+=1
     
    #Now counting the total false negatives:
    for key,category in reference_set.items():
        false_negatives[category]+=1
    
    return true_positives,false_positives,false_negatives


def safe_divide(n,d):
    if n==0 or d ==0:
        return 0.0
    else:
        return float(n)/float(d)


def print_scores(task_name,categories,true_positives,false_positives,false_negatives):
    total_true_positives=0
    total_false_positives=0
    total_false_negatives=0

    for category in categories:

        total_true_positives+=true_positives[category]
        total_false_positives+=false_positives[category]
        total_false_negatives+=false_negatives[category]
        
        logging.debug('True Positives {0},False Positives {1}, False Negatives {2}'.format(true_positives[category],false_positives[category],false_negatives[category]))

        recall= safe_divide(true_positives[category],true_positives[category]+false_negatives[category])
        precision= safe_divide(true_positives[category],true_positives[category]+false_positives[category])
        f_score= safe_divide(2.0*recall*precision,recall+precision)
        print("The support, recall, precision, and f-score for Task {0} : Category {1} is {2},{3},{4},{5}".format(task_name,category,true_positives[category]+false_negatives[category],recall,precision,f_score))

    micro_recall= safe_divide(total_true_positives,total_true_positives+total_false_negatives)
    micro_precision= safe_divide(total_true_positives,total_true_positives+total_false_positives)
    micro_f_score= safe_divide(2.0*micro_recall*micro_precision,micro_recall+micro_precision)
 

    print("The support, recall, precision, and f-score for Task {0} : Micro score is {1},{2},{3},{4}".format(task_name,total_true_positives+total_false_negatives,micro_recall,micro_precision,micro_f_score))
    return 



def get_f_scores(reference_dir,prediction_dir,text_dir,suppress_output=False):

    annotation_tp={categ:0 for categ in ANNOTATION_TYPE}
    annotation_fp={categ:0 for categ in ANNOTATION_TYPE}
    annotation_fn={categ:0 for categ in ANNOTATION_TYPE}

    word_tp={categ:0 for categ in ANNOTATION_TYPE}
    word_fp={categ:0 for categ in ANNOTATION_TYPE}
    word_fn={categ:0 for categ in ANNOTATION_TYPE}

    relation_tp={categ:0 for categ in RELATION_TYPE}
    relation_fp={categ:0 for categ in RELATION_TYPE}
    relation_fn={categ:0 for categ in RELATION_TYPE}

    if not isinstance(reference_dir,str):
        # instead of dirname, a list of files in the refernce dir is provided. Useful for only testing a particular set of files. It is assumed that all files are in the same directory. 
        filename_list=[os.path.basename(fn) for fn in reference_dir]
        reference_dir=os.path.dirname(reference_dir[0])
    else:
        filename_list= os.listdir(reference_dir)

    for filename in filename_list:
        logging.debug('Processing {0}'.format(filename))
        if not filename.endswith('bioc.xml'):
            continue

        assert os.path.isfile(os.path.join(prediction_dir,filename)), 'Prediction file {0} does not exist'.format(os.path.join(prediction_dir,filename))
        assert os.path.isfile(os.path.join(text_dir,filename[:-9])), 'Text file {0} does not exist'.format(os.path.join(text_dir,filename[:-9]))

        annotations,relations,annotation_map,relation_map=read_reference(os.path.join(reference_dir,filename))
        pred_annotations,pred_relations=read_prediction(os.path.join(prediction_dir,filename))

        #Removing the last 9 characters ".bioc.xml" from the filename to get the text file.
        _,word_reference_map=read_word_based_annotations(os.path.join(reference_dir,filename),os.path.join(text_dir,filename[:-9]))
        word_predictions,__=read_word_based_annotations(os.path.join(prediction_dir,filename),os.path.join(text_dir,filename[:-9]))

        atp,afp,afn=generate_match(ANNOTATION_TYPE,annotation_map,pred_annotations)
        annotation_tp={ky:annotation_tp[ky]+atp[ky] for ky in annotation_tp}
        annotation_fp={ky:annotation_fp[ky]+afp[ky] for ky in annotation_fp}
        annotation_fn={ky:annotation_fn[ky]+afn[ky] for ky in annotation_fn}


        wtp,wfp,wfn=generate_match(ANNOTATION_TYPE,word_reference_map ,word_predictions)
        word_tp={ky:word_tp[ky]+wtp[ky] for ky in word_tp}
        word_fp={ky:word_fp[ky]+wfp[ky] for ky in word_fp}
        word_fn={ky:word_fn[ky]+wfn[ky] for ky in word_fn}

        rtp,rfp,rfn=generate_match(RELATION_TYPE,relation_map,pred_relations)
        relation_tp={ky:relation_tp[ky]+rtp[ky] for ky in relation_tp}
        relation_fp={ky:relation_fp[ky]+rfp[ky] for ky in relation_fp}
        relation_fn={ky:relation_fn[ky]+rfn[ky] for ky in relation_fn}
    
    if not suppress_output:
        print_scores("NER-STRICT",ANNOTATION_TYPE,annotation_tp,annotation_fp,annotation_fn)
        print_scores("NER-APPROXIMATE",ANNOTATION_TYPE,word_tp,word_fp,word_fn)
        print_scores("RELATION",RELATION_TYPE,relation_tp,relation_fp,relation_fn)

    

    return annotation_tp,annotation_fp,annotation_fn,relation_tp,relation_fp,relation_fn





if __name__=="__main__":
    # Usage : 
    # For concise output 
    # python bioc_evaluation.py annotation_dir prediction_dir text_dir
    # For verbose output 
    # python bioc_evaluation.py annotation_dir prediction_dir text_dir v


    assert len(sys.argv) >=4, 'Please provide the annotation, prediction and text directories.\npython bioc_evaluation.py annotation_dir prediction_dir text_dir.' 
    ANNOTATION_DIR=sys.argv[1]
    PREDICTION_DIR=sys.argv[2]
    TEXT_DIR=sys.argv[3]


    get_f_scores(ANNOTATION_DIR,PREDICTION_DIR,TEXT_DIR)





