from sklearn.base import BaseEstimator, TransformerMixin


class BaseModel(BaseEstimator, TransformerMixin):
    """Base class for all MLaaS Learn models."""

    def train(self, X, y=None):
        """Fit the model with the training data. Must be overridden in the implementation class.

        Args:
            X (pandas.DataFrame): A data frame containing the texts.
            y (list or pandas.Series): A list of labels for the training data. Will use the 'tag' column of the training data when None is passed. Defaults to None.

        Returns:
            BaseModel: An object of the class.
        """
        return self
    
    
    def fit(self, X, y=None):
        """Fit the model with the training data.

        Args:
            X (pandas.DataFrame): A data frame containing the texts.
            y (list or pandas.Series): A list of labels for the training data. Will use the 'tag' column of the training data when None is passed. Defaults to None.

        Returns:
            BaseModel: An object of the class.
        """
        return self.train(X, y)


    def fit_transform(self, X, y=None):
        """Dummy fit transform for the classifier model to be called from within Pipeline.

        Args:
            X (pandas.DataFrame): A data frame containing the texts.
            y (list or pandas.Series): A list of labels for the training data. Will use the 'tag' column of the training data when None is passed. Defaults to None.

        Returns:
            BaseModel: An object of the class.
        """
        return self.fit(X, y)
    
    
    def transform(self, X):
        """Perform transform operation. Must be overridden in the implementation class.

        Args:
            X (dict or list): A dictionary, list of dictionaries or a list of list of XMLs.

        Returns:
            dict or list: A dictionary of predictions and confidence or a list of dictionaries with predictions and confidences for all pages.
        """
        return X


    def predict(self, X):
        """Perform classification on samples in XMLs.

        Args:
            X (dict or list): A dictionary, list of dictionaries or a list of list of XMLs.

        Returns:
            dict or list: A dictionary of predictions and confidence or a list of dictionaries with predictions and confidences for all pages.
        """
        return self.transform(X)


class BaseDocClassifier(BaseModel):
    """Base class for document classifier."""

    def classify_document(self, xmls):
        """Classify all the pages of a document. Must be overridden in the implementation class.

        Args:
            xmls (list): A list of XMLs for all pages of the document.

        Returns:
            dict: A dictionary containing a list of predictions and a list of confidence for all pages.
        """
        return {'prediction': [], 'confidence': []}


    def transform(self, X):
        """Perform classification on samples in XMLs.

        Args:
            X (dict or list): A dictionary, list of dictionaries or a list of list of XMLs.

        Returns:
            dict or list: A dictionary of predictions and confidence or a list of dictionaries with predictions and confidences for all pages.
        """
        if isinstance(X, dict):
            results = self.classify_document(X['xmls'])
        elif isinstance(X, list) and len(X) > 0:
            if isinstance(X[0], list):
                results = self.classify_document(X[0])
            elif isinstance(X[0], dict):
                results = [self.classify_document(x['xmls']) for x in X]
            elif isinstance(X[0], str):
                results = self.classify_document(X)
        else:
            results = {}

        return results


class BaseInformationExtractor(BaseModel):
    """Base class for information extractors.
    
    Attributes:
        perfrom_transform (bool, optional): Perform transform in fit_transform if True. Defaults to False.
    """
    
    def __init__(self, perfrom_transform=False):
        self.perfrom_transform = perfrom_transform


    def extract(self, word_df, **kwargs):
        """Classify all the pages of a document. Must be overridden in the implementation class.

        Args:
            word_df (pandas.DataFrame): A word data frame comprising all pages of the document.

        Returns:
            list: A list containing a list of predictions and a list of confidence for all pages.
        """
        return word_df


    def fit_transform(self, X, y=None):
        """Train a BaseInformationExtractor model.

        Args:
            X (pandas.DataFrame): The training data frame.
            y (list or pandas.Series): A list of labels for the training data. Will use the 'tag' column of the training data when None is passed. Defaults to None.

        Returns:
            pandas.DataFrame: The transformed data.
        """
        self.fit(X)
        if self.perfrom_transform:
            X = self.transform(X)

        return X


    def transform(self, X):
        """Perform classification on samples in XMLs.

        Args:
            X (list or pandas.DataFrame): A list of data frames or a data frame of test cases.

        Returns:
            pandas.DataFrame or list: A data frame or a list of data frames with extracted entities.
        """
        return [self.extract(x) for x in X] if isinstance(X, list) else self.extract(X)
