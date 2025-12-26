# coding: utf-8

import os
import pandas as pd
import numpy as np
from statistics import median

from .xml_reader import read_xml_from_directory, scale_to_standard, shift_to_standard, convert_word_to_line, get_ratio

from sklearn.base import BaseEstimator, TransformerMixin


class CoordinateMatrixMachine(BaseEstimator, TransformerMixin):
    """A coordinate matrix-based approach that augments human intelligence to learn the structure of the document and use this information to classify the documents and to extract entities from it.
    
    Attributes:
        max_penalty (int, optional): Maximum Penalty is the maximum distance allowed between two keywords. If the distance between the same keywords from two documents is more than this threshold, then the actual distance is substituted by this value. Defaults to 100.
    """

    @staticmethod
    def __calculate_first_value(xml_word, xml_line, token, token_line_id):
        return xml_word[((xml_word['line_left'] == xml_line.loc[token_line_id, 'line_left']) & 
                         (xml_word['line_top'] == xml_line.loc[token_line_id, 'line_top']) & 
                         (xml_word['line_right'] == xml_line.loc[token_line_id, 'line_right']) &
                         (xml_word['line_bottom'] == xml_line.loc[token_line_id, 'line_bottom']) &
                         (xml_word['word'] == token.split()[0]))]


    @staticmethod
    def __calculate_last_value(xml_word, xml_line, token, token_line_id):
        return xml_word[((xml_word['line_left'] == xml_line.loc[token_line_id, 'line_left']) & 
                         (xml_word['line_top'] == xml_line.loc[token_line_id, 'line_top']) & 
                         (xml_word['line_right'] == xml_line.loc[token_line_id, 'line_right']) &
                         (xml_word['line_bottom'] == xml_line.loc[token_line_id, 'line_bottom']) &
                         (xml_word['word'] == token.split()[-1]))]


    @staticmethod
    def __calculate_after_values(xml_word, xml_line, last_value, line_id_value):
        after_value_bot = min(xml_word['word_left'][((xml_word['word_left'] > last_value.iloc[-1]) & 
                                                     (xml_word['word_top'] < xml_line.loc[line_id_value, 'line_top']) & 
                                                     (xml_word['word_bottom'] > xml_line.loc[line_id_value, 'line_top']))], default=1000)

        after_value_top = min(xml_word['word_left'][((xml_word['word_left'] > last_value.iloc[-1]) & 
                                                     (xml_word['word_top'] < xml_line.loc[line_id_value, 'line_bottom']) & 
                                                     (xml_word['word_bottom'] > xml_line.loc[line_id_value, 'line_bottom']))], default=1000)

        after_value_mid = min(xml_word['word_left'][((xml_word['word_left'] > last_value.iloc[-1]) & 
                                                     (xml_word['word_top'] < (xml_line.loc[line_id_value, 'line_bottom'] + xml_line.loc[line_id_value, 'line_top']) / 2) & 
                                                     (xml_word['word_bottom'] > (xml_line.loc[line_id_value, 'line_bottom'] + xml_line.loc[line_id_value, 'line_top']) / 2))], default=1000)
        
        return after_value_bot, after_value_mid, after_value_top


    @staticmethod
    def __encode(xml_word, reference, page=1, resize_by=75):
        xml_line = convert_word_to_line(xml_word)
        
        # Get the coordinates of the labels and values
        ref_info = pd.DataFrame(index=np.arange(reference.shape[0]), 
                                columns=('key', 'label', 'value', 'label_line_left', 'label_line_top', 'label_line_right', 'label_line_bottom',
                                         'value_line_left', 'value_line_top', 'value_line_right', 'value_line_bottom', 'page'))
        
        ref_info['key'] = reference['key']
        ref_info['label'] = reference['label']
        ref_info['value'] = reference['value']
        
        for i in reference.index:
            # Searching for the label 
            label = reference.loc[i, 'label']

            label_line_id = -1
            for j in xml_line.index:
                if (label in xml_line.loc[j, 'word']):
                    label_line_id = j
                    break
            if (label_line_id == -1):
                # The label is not found. (-1,-1,-1,-1) is used as the label coordinate
                ref_info.loc[i, 'label_line_left'] = -1
                ref_info.loc[i, 'label_line_top'] = -1
                ref_info.loc[i, 'label_line_right'] = -1
                ref_info.loc[i, 'label_line_bottom'] = -1            
            else:
                first_value = CoordinateMatrixMachine.__calculate_first_value(xml_word, xml_line, label, label_line_id)['word_left']
                
                if len(first_value) > 0:
                    ref_info.loc[i, 'label_line_left'] = first_value.iloc[0]
                else:
                    ref_info.loc[i, 'label_line_left'] = xml_line.loc[label_line_id, 'line_left']

                last_value = CoordinateMatrixMachine.__calculate_last_value(xml_word, xml_line, label, label_line_id)['word_right']
                
                if len(last_value) > 0:
                    ref_info.loc[i, 'label_line_right'] = last_value.iloc[0]
                else:
                    ref_info.loc[i, 'label_line_right'] = xml_line.loc[label_line_id, 'line_right']
                            
                ref_info.loc[i, 'label_line_top'] = xml_line.loc[label_line_id, 'line_top']            
                ref_info.loc[i, 'label_line_bottom'] = xml_line.loc[label_line_id, 'line_bottom']
            
            # Searching for the value
            value = reference.loc[i, 'value']

            # Finding all the lines that includes the value
            value_id_list = [index for index in xml_line.index if value in xml_line.loc[index, 'word']]
            
            if (len(value_id_list) == 0):
                print(value + ' not found.')
                break

            # Finding the closest line to the label that includes the value 
            dist_list = [abs(ref_info.loc[i, 'label_line_right'] - xml_line.loc[location, 'line_left']) +  abs(ref_info.loc[i, 'label_line_bottom'] - xml_line.loc[location, 'line_top']) for location in value_id_list]
            line_id_value = value_id_list[dist_list.index(min(dist_list))]
                    
            # Revise the start of the line by getting the coordinates of the first value word
            first_value = CoordinateMatrixMachine.__calculate_first_value(xml_word, xml_line, value, line_id_value)['word_left']
            
            if len(first_value) > 0:
                ref_info.loc[i, 'value_line_left'] = first_value.iloc[0]
            else:
                ref_info.loc[i, 'value_line_left'] = xml_line.loc[line_id_value, 'line_left']

            # Revise the end of the line by getting the coordinates of the last value word
            last_value = CoordinateMatrixMachine.__calculate_last_value(xml_word, xml_line, value, line_id_value)['word_right']
            
            if len(last_value) > 0:
                after_value_bot, after_value_mid, after_value_top = CoordinateMatrixMachine.__calculate_after_values(xml_word, xml_line, last_value, line_id_value)
                
                ref_info.loc[i, 'value_line_right'] = last_value.iloc[0] + (min(after_value_bot, after_value_top, after_value_mid) - last_value.iloc[0]) * (resize_by / 100)
            else:
                ref_info.loc[i, 'value_line_right'] = xml_line.loc[line_id_value, 'line_right']
            
            ref_info.loc[i, 'value_line_top'] = xml_line.loc[line_id_value, 'line_top']
            ref_info.loc[i, 'value_line_bottom'] = xml_line.loc[line_id_value, 'line_bottom']
                                
            ref_info.loc[i, 'page'] = page

        return ref_info


    @staticmethod
    def encode(training_data, reference_filename='reference.csv', input_type='page', page=1, resize_by=75, verbose=False):
        """Encodes the raw training data into coordinate matrix.
        
        A static method to create the coordinate matrix from the raw training data.

        Args:
            training_data (str): Fully qualified folder name for the training data.
            reference_filename (str, optional): The csv file with tagged entities. Defaults to 'reference.csv'.
            input_type (str, optional): Consider the entites per block if 'block' is provided. Defaults to 'page'.
            page (int, optional): The page number from where the entities are used. Defaults to 1.
            resize_by (int, optional): The portion of extra white space to be included after the entity value. Defaults to 75.
            verbose (bool, optional): Enables verbose mode if True. Defaults to False.

        Returns:
            pandas.DataFrame: The coordinate matrix for the training data.
        """
        trained_data = []
        for folder in os.scandir(training_data):
            if folder.is_dir():
                for file in os.scandir(folder):
                    if file.is_dir():
                        if verbose:
                            print(file.name)
                        xmls = read_xml_from_directory(file.path)
                        xml_train = scale_to_standard(xmls, standard=1000, smart_cell=False)
                        if input_type == 'block':
                            xml_train = shift_to_standard(xml_train, xmls[0], get_ratio(xmls, standard=1000))
                        key_value = pd.read_csv(file.path + '/' + reference_filename)
                        one_doc = CoordinateMatrixMachine.__encode(xml_train, key_value, page, resize_by)
                        one_doc.insert(0, 'template', folder.name + ': ' + file.name)
                        trained_data.append(one_doc)

        trained_data = pd.concat(trained_data)
        trained_data.columns = ('template', 'key', 'label', 'value', 'label_line_left', 'label_line_top', 'label_line_right', 'label_line_bottom', 'value_line_left', 'value_line_top', 'value_line_right', 'value_line_bottom', 'page')
        trained_data = trained_data.reset_index(drop=True)

        return trained_data


    def __init__(self, max_penalty=100):
        """Instantiate an instance of the class CoordinateMatrixMachine.

        Args:
            max_penalty (int, optional): Maximum Penalty is the maximum distance allowed between two keywords. If the distance between the same keywords from two documents is more than this threshold, then the actual distance is substituted by this value. Defaults to 100.
        """
        self.max_penalty = max_penalty
        self.trained_data = None


    def __classify(self, X):
        """Perform classification on samples in X.

        Args:
            X (pandas.DataFrame): A dataframe of test cases created using mlaas_py_utilties.xml_reader.parse_xmls.

        Returns:
            list: A list containing prediction, confidence and the number of matched labels.
        """
        xml_word = X.copy()
        trained_data = self.trained_data.copy()
        all_labels = trained_data.label.unique()
        xml_line = convert_word_to_line(xml_word)

        label_info = pd.DataFrame(columns=['label', 'label_left', 'label_top'])
        label_info['label'] = all_labels

        for i in label_info.index:
            label = label_info.loc[i,'label']

            label_id_list = [index for index in xml_line.index if label in xml_line.loc[index, 'word']]

            if len(label_id_list) == 0:
                continue

            label_line_id = label_id_list[0]

            first_value = self.__calculate_first_value(xml_word, xml_line, label, label_line_id)

            if len(first_value) > 0:
                label_info.loc[i, 'label_left'] = first_value['word_left'].iloc[0]
                label_info.loc[i, 'label_top'] = first_value['word_top'].iloc[0]            
            else:
                label_info.loc[i, 'label_left'] = xml_line.loc[label_line_id, 'line_left'] 
                label_info.loc[i, 'label_top'] = xml_line.loc[label_line_id, 'line_top'] 

        trained_data = pd.merge(trained_data, label_info, on='label')
        trained_data['distance'] = abs(trained_data['label_line_left'] - trained_data['label_left']) + abs(trained_data['label_line_top'] - trained_data['label_top'])
        trained_data['distance'] = trained_data['distance'].apply(lambda x: np.nanmin([x, self.max_penalty]))
        trained_data.drop(['label_left', 'label_top'], axis=1, inplace=True)
        file_dist = trained_data.loc[trained_data['label_line_left'] != -1, ['template', 'distance']]
        file_dist = file_dist.groupby('template')['distance'].mean()
        n_labels = len(trained_data.loc[((trained_data['template'] == file_dist.idxmin()) & (~trained_data['key'].apply(lambda x: x.startswith('header_'))) & (trained_data['distance'] < self.max_penalty)), ]) 

        return [file_dist.idxmin(), (1 - (file_dist.min() / self.max_penalty) ** 2), n_labels]


    def __extract(self, X, template, page):
        """Extract entities from samples in X.

        Args:
            X (pandas.DataFrame): A dataframe of test cases created using mlaas_py_utilties.xml_reader.parse_xmls.
            template (str): The classification result from predict method.
            page (int): The page number from which entities to be extracted.

        Returns:
            pandas.DataFrame: A dataframe containing the extraction results.
        """
        xml_word = X.copy()
        trained_data = self.trained_data.copy()

        trained_data = trained_data[-trained_data['key'].str.startswith('cbx_')]
        trained_data = trained_data[-trained_data['key'].str.startswith('header_')]
        trained_data = trained_data[-trained_data['key'].str.startswith('issuing_bank')]

        ref_info = trained_data.loc[trained_data['template'] == template, :].copy()
        ref_info = ref_info.drop('template', axis=1)
        ref_info = ref_info.reset_index(drop=True)
        ref_info = ref_info.infer_objects(copy=False).fillna('')

        # Loading the target file
        xml_line = convert_word_to_line(xml_word)

        new_doc_info = pd.DataFrame(index=np.arange(ref_info.shape[0]),
                                    columns=('template', 'key', 'label', 'value', 'label_line_left', 'label_line_top', 'label_line_right', 'label_line_bottom',
                                             'value_line_left', 'value_line_top', 'value_line_right', 'value_line_bottom', 'page', 'confidence'))

        new_doc_info['key'] = ref_info['key']
        new_doc_info['label'] = ref_info['label']

        # Finding the labels' coordinates
        for i in ref_info.index:
            if ref_info.loc[i, 'label_line_left'] == -1:
                new_doc_info.loc[i, 'label_line_left'] = -1
                new_doc_info.loc[i, 'label_line_top'] = -1
                new_doc_info.loc[i, 'label_line_right'] = -1
                new_doc_info.loc[i, 'label_line_bottom'] = -1
                continue     

            label = ref_info.loc[i, 'label']
            label_id_list = [index for index in xml_line.index if label in xml_line.loc[index, 'word']]
            
            if len(label_id_list) == 0:
                print('Label: "' + label + '" not found.')
                continue
            
            # Finding the closest line to the reference label that includes the label 
            dist_list = [abs(ref_info.loc[i, 'label_line_left'] - xml_line.loc[location, 'line_left']) +  abs(ref_info.loc[i, 'label_line_top'] - xml_line.loc[location, 'line_top']) for location in label_id_list]
            label_line_id = label_id_list[dist_list.index(min(dist_list))]

            first_value = self.__calculate_first_value(xml_word, xml_line, label, label_line_id)['word_left']
        
            if len(first_value) > 0:
                new_doc_info.loc[i, 'label_line_left'] = first_value.iloc[0]
            else:
                new_doc_info.loc[i, 'label_line_left'] = xml_line.loc[label_line_id, 'line_left']

            last_value = self.__calculate_last_value(xml_word, xml_line, label, label_line_id)['word_right']
            
            if len(last_value) > 0:
                new_doc_info.loc[i, 'label_line_right'] = last_value.iloc[0]
            else:
                new_doc_info.loc[i, 'label_line_right'] = xml_line.loc[label_line_id, 'line_right']        
            
            new_doc_info.loc[i, 'label_line_top'] = xml_line.loc[label_line_id, 'line_top']
            new_doc_info.loc[i, 'label_line_bottom'] = xml_line.loc[label_line_id, 'line_bottom']

        # Finding the values' coordinates
        new_doc_info['label_line_left'] = pd.Series([-1 if np.isnan(x) else float(x) for x in new_doc_info['label_line_left']])
        nonfix_ids = (new_doc_info['label_line_left'] != -1)
        for i in ref_info.index:
            # All the values' coordinates in the info file are used to estimate the value'cordinates of the target sample
            try:
                left_est = median(new_doc_info.loc[nonfix_ids, 'label_line_left'] + ref_info.loc[i, 'value_line_left'] - ref_info.loc[nonfix_ids, 'label_line_left'])
                top_est = min(new_doc_info.loc[nonfix_ids, 'label_line_top'] + ref_info.loc[i, 'value_line_top'] - ref_info.loc[nonfix_ids, 'label_line_top'])
            except:
                print('Entity not found for "' + str(new_doc_info.loc[i, 'key']) + '".')
                new_doc_info.loc[i, 'value'] = ''
                continue

            right_est = left_est + ref_info.loc[i, 'value_line_right'] - ref_info.loc[i, 'value_line_left']
            bottom_est = top_est + ref_info.loc[i, 'value_line_bottom'] - ref_info.loc[i, 'value_line_top']
            mid_horizontal = (top_est + bottom_est) / 2
            padding_horizontal = (bottom_est - top_est) / 2

            # Extracting all words in the estimated line
            overlap_lines = xml_word[((xml_word['word_top'] <= mid_horizontal + padding_horizontal) & 
                                      (xml_word['word_bottom'] >= mid_horizontal - padding_horizontal) & 
                                      (xml_word['word_right'] >= left_est) & 
                                      (xml_word['word_left'] <= right_est))]

            if overlap_lines.shape[0] == 0:
                print('Entity not found for "' + str(new_doc_info.loc[i, 'key']) + '".')
                new_doc_info.loc[i, 'value'] = ''
                new_doc_info.loc[i, 'value_line_left'] = left_est
                new_doc_info.loc[i, 'value_line_top'] = top_est
                new_doc_info.loc[i, 'value_line_right'] = right_est
                new_doc_info.loc[i, 'value_line_bottom'] = bottom_est
                new_doc_info.loc[i, 'confidence'] = -1
            else:
                # Concatenating all the extracted words
                new_doc_info.loc[i, 'value'] = ' '.join(overlap_lines['word'])
                if left_est < 0:
                    new_doc_info.loc[i, 'value_line_left'] = min(overlap_lines['word_left'])
                else:
                    new_doc_info.loc[i, 'value_line_left'] = min(pd.concat([overlap_lines['word_left'], pd.Series([left_est])]))

                if top_est < 0:
                    new_doc_info.loc[i, 'value_line_top'] = min(overlap_lines['word_top'])
                else:
                    new_doc_info.loc[i, 'value_line_top'] = min(pd.concat([overlap_lines['word_top'], pd.Series([top_est])]))

                new_doc_info.loc[i, 'value_line_right'] = max(pd.concat([overlap_lines['word_right'], pd.Series([right_est])]))
                new_doc_info.loc[i, 'value_line_bottom'] = max(pd.concat([overlap_lines['word_bottom'], pd.Series([bottom_est])]))
                new_doc_info.loc[i, 'confidence'] = 1 - abs(mid_horizontal - median(overlap_lines['word_bottom'] + overlap_lines['word_top']) / 2) / (2 * padding_horizontal)

            new_doc_info.loc[i, 'page'] = page

        if len(new_doc_info) > 0:
            new_doc_info.loc[new_doc_info['confidence'] == -1, 'confidence'] = new_doc_info.loc[new_doc_info['confidence'] == -1, 'confidence'].count() / len(new_doc_info)

        return new_doc_info


    def train(self, X):
        """Fit the CM2 model according to the given training data. A previously fitted model will add the new templates to the existing model. If a duplicate template is added, model will discard the old data for the template and retain only the new data of the template.
        
        CAUTION: In the event, new template contains different keys as opposed to the original template, only the duplicate keys will be removed, and the model will have part of newer and older template, and hence it is advisable in those event to train the model with full set of training data.

        Args:
            X (pandas.DataFrame): The coordinate matrix for the training data.

        Returns:
            CoordinateMatrixMachine: The model is trained/re-trained.
        """
        if self.trained_data is None:
            self.trained_data = X
        else:
            self.trained_data = pd.concat([self.trained_data, X], ignore_index=True)
            self.trained_data.drop_duplicates(['template', 'key'], keep='last', ignore_index=True, inplace=True)

        return self


    def fit(self, X):
        """Fit the CM2 model according to the given training data. A previously fitted model will add the new templates to the existing model. If a duplicate template is added, model will discard the old data for the template and retain only the new data of the template.
        
        CAUTION: In the event, new template contains different keys as opposed to the original template, only the duplicate keys will be removed, and the model will have part of newer and older template, and hence it is advisable in those event to train the model with full set of training data.

        Args:
            X (pandas.DataFrame): The coordinate matrix for the training data.

        Returns:
            CoordinateMatrixMachine: The model is trained/re-trained.
        """
        return self.train(X)


    def transform(self, X):
        """A dummy transform method.

        Args:
            X (object): A dummy object

        Returns:
            CoordinateMatrixMachine: The model already trained.
        """
        return self


    def fit_transform(self, X):
        """A dummy fit_transform method.

        Args:
            X (object): A dummy object

        Returns:
            CoordinateMatrixMachine: The model already trained.
        """
        return self.fit(X)


    def predict(self, X):
        """Perform classification on samples in X.

        Args:
            X (pandas.DataFrame or list): A data frame or list of data frames of test cases created using mlaas_py_utilties.xml_reader.parse_xmls.

        Returns:
            list: A list or list of lists containing prediction, confidence and the number of matched labels.
        """
        return [self.__classify(x) for x in X] if isinstance(X, list) else self.__classify(X)


    def extract(self, X, template, page):
        """Extract entities from samples in X.

        Args:
            X (pandas.DataFrame or list): A data frame or list of data frames of test cases created using mlaas_py_utilties.xml_reader.parse_xmls.
            template (str or list): The classification result or list of classification results from predict method.
            page (int or list): The page number or list of page numbers from which entities to be extracted.

        Returns:
            pandas.DataFrame or list: A data frame or list of data frames containing the extraction results.
        """
        return [self.__extract(X[i], template[i], page[i]) for i, _ in enumerate(X)] if isinstance(X, list) else self.__extract(X, template, page)
