from .xml_reader import scale_to_standard, shift_to_standard, get_ratio
from .model import BaseDocClassifier
from .cm2 import CoordinateMatrixMachine
from .ienum import IndexableEnum


class DocumentClass(IndexableEnum):
    """An enumeration of standard document classes."""
    AdditionalPageGeneric = 'Additional Page:'
    AdditionalPage = 'Additional Page: Previous Document'
    EmptyPage = 'Empty Page: No Data'
    UnknownDocument = 'Unknown Document: Unknown Page'
    UnknownBank = 'Unknown Bank: Unknown Template'
   

class StructuredDocumentClassifier(BaseDocClassifier):
    """Base class for fixed structured document classifier.

    Attributes:
        max_penalty (int): Maximum Penalty for CoordinateMatrixMachine. Maximum Penalty is the maximum distance allowed between two keywords. If the distance between the same keywords from two documents is more than this threshold, then the actual distance is substituted by this value.
        min_confidence (float): Minimum confidence to consider for a class, below which the class becomes 'Unknown Template' or 'Additional Pages'.
        max_confidence (float): Maximum confidence to consider for a class, above which the minimum labels are ignored.
        min_labels (int): Minimum number of labels that needs to match to be considered for a template to match a trained template.
        special (str, optional): Need any use case specific language. Currently supported are: 'OFI' and None. Defaults to None.
        debug (bool, optional): Verbose mode if True. Defaults to False.
    """

    def __init__(self, max_penalty, min_confidence, max_confidence, min_labels, special=None, debug=False):
        self.max_penalty = max_penalty
        self.cm2 = CoordinateMatrixMachine(self.max_penalty)
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.min_labels = min_labels
        self.special = special
        self.debug = debug


    def train(self, X, y=None):
        """Fit the model with the training data.

        Args:
            X (pandas.DataFrame): A data frame containing the texts.
            y (list or pandas.Series): A list of labels for the training data. Will use the 'tag' column of the training data when None is passed. Defaults to None.

        Returns:
            StructuredDocumentClassifier: An object of the class.
        """
        self.cm2.fit(X)
        return self


    def classify_page(self, xml_word, page=1):
        """Classify a single page of a document.

        Args:
            xml_word (pandas.DataFrame): A data frame representation of a single page of OCR XMLs, generated using scale_to_standard funciton of the mlaas_py_utilities.xml_reader module.
            page (int, optional): The page number to process. Defaults to 1.

        Returns:
            list: A list containing the prediction and confidence for that page of the document.
        """
        if len(xml_word) == 0:
            return DocumentClass.EmptyPage.value, 0.99

        prediction, confidence, n_labels = self.cm2.predict(xml_word)

        if self.debug:
            logger.info([prediction, confidence, n_labels])

        if confidence <= self.min_confidence or n_labels < self.min_labels:
            if not (confidence >= self.max_confidence and n_labels > 0):
                if page == 1:
                    if self.special is None:
                        return DocumentClass.UnknownDocument.value, confidence
                    elif self.special.upper() == 'OFI':
                        return DocumentClass.UnknownBank.value, confidence                        
                else:
                    return DocumentClass.AdditionalPage.value, confidence
        
        return prediction, confidence


    def classify_document(self, xmls):
        """Classify all pages of a document.

        Args:
            xmls (list): A list of OCR XMLs.

        Returns:
            dict: A dictionary containing a list of predictions and a list of confidences for all pages of the document.
        """
        prediction = []
        confidence = []

        for index, xml in enumerate(xmls):
            xml_df = scale_to_standard([xml], smart_cell=False, standard=1000)
            xml_df = shift_to_standard(xml_df, xml, get_ratio([xml], standard=1000)) if self.special is None else xml_df
            pred, conf = self.classify_page(xml_df, index + 1)

            prediction.append(pred)
            confidence.append(conf)

        return {'prediction': prediction, 'confidence': confidence}
