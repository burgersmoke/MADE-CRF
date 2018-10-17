# Very simple, Annotation classes agnostic of annotation format    
# (i.e. doesn't care if it came from BRAT, BIOC, eHost, etc)    
    
# this class encapsulates all data related to a span (text sequence) annotation    
# including the text it "covers" and the type (i.e. "concept") of the annotation    
class Annotation(object):    
    def __init__(self):    
        self.start_index = -1    
        self.end_index = -1    
        self.type = ''    
        self.spanned_text = ''

    def __repr__(self):
        return '[{0}], {1}:{2}, type=[{3}]'.format(self.spanned_text, self.start_index, self.end_index, self.type)
    
    # adding this so that pyConText's HTML markup can work seamlessly    
    def getSpan(self):    
        return (self.start_index, self.end_index)    
    
    def getCategory(self):    
        # pyConText graph objects actually expect a list here    
        return [self.type]
        
# this class encapsulates all data for a document which has been annotated_doc_map
# this includes the original text, its annotations and also
class AnnotatedDocument(object):
    def __init__(self):
        self.filename = ''
        self.text = ''
        self.annotations = []
        # NOTE : This "positive_label" relates to positive/possible cases of pneumonia
        self.positive_label = -1
        self.tokenized_document = None