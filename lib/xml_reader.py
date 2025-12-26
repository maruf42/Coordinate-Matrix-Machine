# coding: utf-8

"""A collection of utilities to handle Datacap OCR XMLs."""

import pandas as pd
import xml.etree.ElementTree as et
from lxml import etree
import os, re, json, xmltodict


CHAR_LIST = list(r'1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ<>%&@!*,;$"/.\' :()-•£')

def calculate_non_ascii_ratio(text):
    """Returns: the non-ASCII ratio in the text.
    
    Args:
        text (str): Any text.

    Returns:
        float: The ratio of non-ASCII charaters in the text.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> calculate_non_ascii_ratio(xmls[0])
        0.11087953313880784
    """
    if isinstance(text, list):
        return [calculate_non_ascii_ratio(a) for a in text]
    else:
        s = list(text)
    counter = 0.00
    len_c =  len([c for c in s if c != ' '])

    for c in s:
        if c not in CHAR_LIST:
            counter = counter + 1

    # Avoid the only character is a space
    # The calculation ratio does not include the space
    if len_c > 0:
        return counter / len_c
    else:
        return -10


def convert_word_to_line(df_xml):
    """Converts the DataCap XML DataFrame containing words into lines.
    
    Args:
        df_xml (pandas.DataFrame): A dataframe created by parse_xml(..., output='word'), containing word content.
    
    Returns:
        pandas.DataFrame: A converted dataframe containing line content.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> xml_word = parse_xmls(xmls, output='word')
        >>> xml_line = convert_word_to_line(xml_word)
    """
    df_xml = df_xml.copy()
    df_xml['word_order'] = df_xml.groupby(['block_left', 'block_top', 'block_right', 'block_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'page']).cumcount() + 1

    sent_df = pd.DataFrame(df_xml.sort_index().sort_values(by='word_order').groupby(['block_type', 'table_pos', 'block_left', 'block_top', 'block_right', 'block_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'page'])['word'].apply(' '.join).reset_index())
    sent_df.sort_values(by=['page', 'line_top', 'line_left'], inplace=True)
    sent_df.reset_index(drop=True, inplace=True)

    sent_df = sent_df[['block_type', 'table_pos', 'word', 'line_left', 'line_top', 'line_right', 'line_bottom', 'block_left', 'block_top', 'block_right', 'block_bottom', 'page']]
    sent_df['non_ascii_ratio'] = sent_df['word'].apply(calculate_non_ascii_ratio) if len(sent_df) > 0 else 0

    return sent_df


def parse_xml(xml_string, page=1, output='line', smart_cell=True):
    """Parse the DataCap XML file into a dataframe containing either words or lines.
    
    Args:
        xml_string (str): A string containing XML object
        page (int, optional): Page number to convert to dataframe. Defaults to 1.
        output (str, optional): The resultant dataframe contains lines if 'line' or 'l', otherwise words. Defaults to 'line'.
        smart_cell (bool, optional): To keep tabular data same as block, place <Table> data in the hierarchy of Table, Row, Cell, Word if True, otherwise Table, Cell, Line, Word. Defaults to True.
    
    Returns:
        pandas.DataFrame: A dataframe containing the word or line content.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> xml_line = parse_xml(xmls[0], output='line')
    """
    #  Define the potential data columns
    df_col = ('block_type', 'table_pos', 'block_pos', 'line_pos', 'word_pos', 'word')
    if output == 'line' or output == 'l':
        df_col0 = ('block_type', 'table_pos', 'block_left', 'block_top', 'block_right', 'block_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'word', 'page')
    else:
        df_col0 = ('block_type', 'table_pos', 'block_left', 'block_top', 'block_right', 'block_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'word_left', 'word_top', 'word_right', 'word_bottom', 'word', 'page')

    try:
        #  This part of the code will go through the xml root and children to extract the interested elements in the root
        #  The interest areas will be block position, line position, word position, word contents.

        if xml_string.count('<') < 5:
            return pd.DataFrame(columns=list(df_col0).append('non_ascii_ratio'))

        xtree = et.ElementTree(et.fromstring(xml_string, parser=etree.XMLParser(recover=True)))
        xroot = xtree.getroot()

        xml_list = []
        table_list = []
        
        #  Extract the block
        for child in xroot.findall('Block'):
            for grandchildren in list(child):
                for great_grandchildren in list(grandchildren):
                    if len(great_grandchildren.attrib) > 0:
                        xml_list.append([
                                            str('B'),
                                            str(''),
                                            str(child.attrib['pos']),
                                            str(grandchildren.attrib['pos']),
                                            str(great_grandchildren.attrib['pos']),
                                            str(great_grandchildren.attrib['v'])
                                        ])
                        
        #  Extract the title          
        for child in xroot.findall('Title'):
            for grandchildren in list(child):
                for great_grandchildren in list(grandchildren):
                    if len(great_grandchildren.attrib) > 0:
                        xml_list.append([
                                            str('B'),
                                            str(''),
                                            str(child.attrib['pos']),
                                            str(grandchildren.attrib['pos']),
                                            str(great_grandchildren.attrib['pos']),
                                            str(great_grandchildren.attrib['v'])
                                        ])

        #  Extract the footer
        for child in xroot.findall('Footer'):
            for grandchildren in list(child):
                for great_grandchildren in list(grandchildren):
                    if len(great_grandchildren.attrib) > 0:
                        xml_list.append([
                                            str('F'),
                                            str(''),
                                            str(child.attrib['pos']),
                                            str(grandchildren.attrib['pos']),
                                            str(great_grandchildren.attrib['pos']),
                                            str(great_grandchildren.attrib['v'])
                                        ])
        
        #  If it is "Table", loop into the rows, cells, lines and words.
        if smart_cell:
            if len(xroot.findall('Table')) > 0:
                for table in xroot.findall('Table'):
                    for row in list(table):
                        for cell in list(row):
                            for L in list(cell):
                                for W in list(L): 
                                    if len(W) > 0:
                                        table_list.append ([  
                                                                str('T'),
                                                                str(table.attrib['pos']),
                                                                str(row.attrib['pos']), 
                                                                str(cell.attrib['pos']),
                                                                str(W.attrib['pos']) ,
                                                                str(W.attrib['v'])
                                                            ])
        else:
            if len(xroot.findall('Table')) > 0:
                for table in xroot.findall('Table'):
                    for row in list(table):
                        for cell in list(row):
                            for L in list(cell):
                                for W in list(L): 
                                    if len(W) > 0:
                                        table_list.append ([  
                                                                str('T'),
                                                                str(table.attrib['pos']),
                                                                str(cell.attrib['pos']), 
                                                                str(L.attrib['pos']),
                                                                str(W.attrib['pos']) ,
                                                                str(W.attrib['v'])
                                                            ])
    
        # Extract the table data into a dataframe
        df_table = pd.DataFrame(data=table_list, columns=df_col)
        df_table.drop_duplicates(subset=None, keep='first', inplace=True)
        
        # Extract the xml data into a dataframe
        df_xml = pd.DataFrame(data=xml_list, columns=df_col)
        df_xml.drop_duplicates(subset=None, keep='first', inplace=True)

        # Combine the table and block
        df_xml = pd.concat([df_xml, df_table], ignore_index=True, sort=True)

        # Split the word position into separate columns
        word_pos_df = df_xml['word_pos'].str.split(',', expand=True)
        word_pos_df.columns = ('word_left', 'word_top', 'word_right', 'word_bottom')
        word_pos_df = word_pos_df.apply(pd.to_numeric, errors='coerce')

        # Split the line position into separate columns
        line_pos_df = df_xml['line_pos'].str.split(',', expand=True)
        line_pos_df.columns = ('line_left', 'line_top', 'line_right', 'line_bottom')
        line_pos_df = line_pos_df.apply(pd.to_numeric, errors='coerce')
        
        # Split the block position into separate columns
        block_pos_df = df_xml['block_pos'].str.split(',', expand=True)
        block_pos_df.columns = ('block_left', 'block_top', 'block_right', 'block_bottom')
        block_pos_df = block_pos_df.apply(pd.to_numeric, errors='coerce')

        # Combine back into a new dataframe
        wordpos_df = pd.concat([df_xml, word_pos_df], axis=1)
        wordpos_df = pd.concat([wordpos_df, line_pos_df], axis=1)
        word_linepos_df = pd.concat([wordpos_df, block_pos_df], axis=1)

        word_linepos_df.sort_values(by=['line_top', 'line_left'], inplace=True)
        word_linepos_df.reset_index(drop=True, inplace=True)
        return_df = word_linepos_df.drop(['block_pos', 'line_pos', 'word_pos'], axis=1)
        temp_df = {}
        temp_df['page'] = page
        temp_df['non_ascii_ratio'] = return_df['word'].apply(calculate_non_ascii_ratio) if len(return_df) > 0 else 0
        return_df = pd.concat([return_df, pd.DataFrame(temp_df)], axis=1)

        if output == 'word' or output == 'w':
            return return_df
        else:
            return convert_word_to_line(return_df)
    except ValueError:
        return pd.DataFrame(columns=list(df_col0).append('non_ascii_ratio'))


def check_ocr_quality_xml(xmls):
    """Check the OCR quality based on non-ASCII ratio from DataCap XML files.
    
    Args:
        xmls (list): A list of strings containing XML objects.
    
    Returns:
        float: A floating point value between 0 and 100 defining the quality of the OCR.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> check_ocr_quality_xml(xmls)
        96.71331926058483
    """
    xml_df = parse_xmls(xmls)
    return check_ocr_quality_df(xml_df)


def check_ocr_quality_df(xml_df):
    """Check the OCR quality based on non-ASCII ratio from converted dataframe.
    
    Args:
        xml_df (pandas.DataFrame): A dataframe for XML object generated through parse_xmls.
    
    Returns:
        float: A floating point value between 0 and 100 defining the quality of the OCR.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> xml_word = parse_xmls(xmls, output='word')
        >>> check_ocr_quality_df(xml_word)
        94.03891525399905
    """
    return (1 - (xml_df['non_ascii_ratio'].sum()/xml_df.shape[0])) * 100


def get_element(xml_string, search_tag, page_tag='Page'):
    """Get the value for an attibute from an XML element.
    
    Args:
        xml_string (str): A string containing XML object.
        search_tag (str): A string containing the attribute/tag name.
        page_tag (str, optional): A string containing the element name. Defaults to 'Page'.
    
    Returns:
        str: A string containing the value of the given attibute of the element.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> get_element(xmls[0], 'pos')
        '0,0,2504,3525'
    """
    xtree = et.ElementTree(et.fromstring(xml_string))
    xroot = xtree.getroot()
    for page_info in xroot.iter(page_tag):
        try:
            return page_info.attrib[search_tag]
        except:
            return ''
    

def parse_location(string):
    """Convert the pos attribute of a DataCap XML into a list.
    
    Args:
        string (str): A string containing the value of the pos attribute.
    
    Returns:
        list: A list containing [left, top, right, bottom] for the pos attribute.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> location = get_element(xmls[0], 'pos')
        >>> parse_location(location)
        [0.0, 0.0, 2504.0, 3525.0]
    """
    pos = string.split(',')
    pos = [float(pos[i]) for i in range(len(pos))]

    return pos


def is_landscape(xml_string):
    """To determine whether a given XML is from a landscape page orientation or not.
    
    Args:
        xml_string (str): A string containing XML object.
    
    Returns:
        bool: The given XML is a landscape page or not.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> is_landscape(xmls[0])
        False
    """   
    page_position_info = parse_location(get_element(xml_string, 'pos'))

    page_w = page_position_info[2] - page_position_info[0]
    page_h = page_position_info[3] - page_position_info[1]

    if page_h < page_w:
        return True
    else:
        return False


def _assign_location(a, x, cols, rounding=False):
    """Assign new location while applying scaling to a dataframe from OCR data.
    
    Args:
        a (pandas.DataFrame): A dataframe for XML object generated through parse_xmls.
        x (pandas.DataFrame): A dataframe containing the new location values.
        cols (list): A list of column names where location data resides in both dataframes.
        rounding (bool, optional): A boolean value to determine whether location values need to be rounded or not. Defaults to False.
    
    Returns:
        pandas.DataFrame: A dataframe for XML object with new location data attached.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> cols = ['line_left', 'line_top', 'line_right', 'line_bottom', 'block_left', 'block_top', 'block_right', 'block_bottom']
        >>> ratio = get_ratio(xmls, 1000)
        >>> xml_word = parse_xmls(xmls, output='word')
        >>> scaled_xml_word = xml_word.loc[:, cols] * ratio
        >>> scaled_xml_word = _assign_location(xml_word, scaled_xml_word, cols, False)
    """ 
    b = a.copy()
    
    if rounding:
        for col in cols:
            b[col] = x[col].dropna().apply(round)
    else:
        for col in cols:
            b[col] = x[col]
    
    return b


def get_ratio(xmls, standard=1000):
    """Generate the pixel ratio of a DataCap XML page against a standardised width.
    
    Args:
        xmls (list): A list of strings containing XML object.
        standard (float, optional): A width against with the pixel ratio is generated. Defaults to 1000.
    
    Returns:
        float: A width ratio between the standard and the orginal page.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> get_ratio(xmls)
        0.3993610223642173
    """ 
    pos = parse_location(get_element(xmls[0], 'pos'))
    ratio = standard/(pos[2] - pos[0])

    return ratio


def scale_to_standard(xmls, xml_word=None, standard=1000, smart_cell=True, cols=['word_left', 'word_top', 'word_right', 'word_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'block_left', 'block_top', 'block_right', 'block_bottom']):
    """Convert the position info of the XML data towards the given standard. This mimics the effect of rescaling the XML data.
    
    Args:
        xmls (list): A list of strings containing XML object.
        xml_word (pandas.DataFrame): A data frame for XML object generated through parse_xmls. If no data frame is provided, the function will craete one using parser_xml. Defaults to None.
        standard (float, optional): A width against with the pixel ratio is generated. Defaults to 1000.
        smart_cell (bool, optional): To keep tabular data same as block, place <Table> data in the hierarchy of Table, Row, Cell, Word if True, otherwise Table, Cell, Line, Word. Defaults to True.
        cols (list, optional): A list of position columns to be converted. Defaults to all columns, i.e., ['word_left', 'word_top', 'word_right', 'word_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'block_left', 'block_top', 'block_right', 'block_bottom'].
    
    Returns:
        pandas.DataFrame: A dataframe for XML object with position info on the standardised scale.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> scaled_xml_word = scale_to_standard(xmls)
    """ 
    ratio = get_ratio(xmls, standard)
    xml_word = parse_xmls(xmls, output='w', smart_cell=smart_cell) if xml_word is None else xml_word
    scaled_xml_word = xml_word.loc[:, cols] * ratio
    scaled_xml_word = _assign_location(xml_word, scaled_xml_word, cols, False)

    return scaled_xml_word


def scale_to_original(scaled_xml_word, xmls, standard=1000, cols=['word_left', 'word_top', 'word_right', 'word_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'block_left', 'block_top', 'block_right', 'block_bottom']):
    """Convert the position info of the XML data towards the original scale. This mimics the effect of rescaling the XML data.
    
    Args:
        scaled_xml_word (pandas.DataFrame): A data frame for XML object generated through scale_to_standard.
        xmls (list): A list of strings containing XML object.
        standard (float, optional): A width against with the pixel ratio is generated. Defaults to 1000.
        cols (list, optional): A list of position columns to be converted. Defaults to all columns, i.e., ['word_left', 'word_top', 'word_right', 'word_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'block_left', 'block_top', 'block_right', 'block_bottom'].
    
    Returns:
        pandas.DataFrame: A data frame for XML object with position info on the original scale.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> scaled_xml_word = scale_to_standard(xmls)
        >>> xml_word = scale_to_original(scaled_xml_word, xmls)
    """ 
    ratio = get_ratio(xmls, standard)

    if ratio != 0:
        xml_word = scaled_xml_word.loc[:, cols] / ratio
        xml_word = _assign_location(scaled_xml_word, xml_word, cols, True)
        xml_word[cols] = xml_word[cols].astype('Int32')
    else:
        xml_word = scaled_xml_word
        logger.error('Ratio cannot be zero.')
        
    return xml_word


def shift_to_standard(xml_word, xmls, ratio=1, cols=['block_type', 'table_pos', 'word', 'page', 'non_ascii_ratio'], cols_x=['word_left', 'word_right', 'line_left', 'line_right', 'block_left', 'block_right'], cols_y=['word_top', 'word_bottom', 'line_top', 'line_bottom', 'block_top', 'block_bottom']):
    """Shift the position info of the XML data towards a [0, 0] starting point. This mimics the effect of repositioning the XML data.
    
    Args:
        xml_word (pandas.DataFrame): A dataframe created by parse_xml(..., output='word'), containing word content.
        xmls (list): A list of strings containing XML object.
        ratio (float, optional): A ratio to be applied on how much shifting is required. Defaults to 1.
        cols (list, optional): A list of columns that doesn't requires shifting. Defaults to ['block_type', 'table_pos', 'word', 'page', 'non_ascii_ratio'].
        cols_x (list, optional): A list of position columns which contains positions relevant to x/horizontal-axis. Defaults to ['word_left', 'word_right', 'line_left', 'line_right', 'block_left', 'block_right'].
        cols_y (list, optional): A list of position columns which contains positions relevant to y/vertical-axis. Defaults to ['word_top', 'word_bottom', 'line_top', 'line_bottom', 'block_top', 'block_bottom'].
    
    Returns:
        pandas.DataFrame: A dataframe for XML object with position info towards the 0, 0 shifted position.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> xml_word = parse_xmls(xmls, output='word')
        >>> shifted_xml_word = shift_to_standard(xml_word, xmls)
    """ 
    l, t, r, b = get_element(xmls, 'pos').split(',')
    l = float(l) * ratio
    t = float(t) * ratio

    start = xml_word[cols]
    scaled_xml_word1 = xml_word.loc[:, cols_x] - l
    scaled_xml_word2 = xml_word.loc[:, cols_y] - t

    scaled_xml_word = pd.concat([start, scaled_xml_word1, scaled_xml_word2], axis=1, sort=False)
    scaled_xml_word = scaled_xml_word[xml_word.columns.to_list()]

    return scaled_xml_word


def shift_to_original(scaled_xml_word, xmls, ratio=1, cols=['block_type', 'table_pos', 'word', 'page', 'non_ascii_ratio'], cols_x=['word_left', 'word_right', 'line_left', 'line_right', 'block_left', 'block_right'], cols_y=['word_top', 'word_bottom', 'line_top', 'line_bottom', 'block_top', 'block_bottom']):
    """Shift the position info of the XML data towards the original starting point. This mimics the effect of repositioning the XML data.
    
    Args:
        scaled_xml_word (pandas.DataFrame): A dataframe created by shift_to_standard, containing word content.
        xmls (list): A list of strings containing XML object.
        ratio (float, optional): A ratio to be applied on how much shifting is required. Defaults to 1.
        cols (list, optional): A list of columns that doesn't requires shifting. Defaults to ['block_type', 'table_pos', 'word', 'page', 'non_ascii_ratio'].
        cols_x (list, optional): A list of position columns which contains positions relevant to x/horizontal-axis. Defaults to ['word_left', 'word_right', 'line_left', 'line_right', 'block_left', 'block_right'].
        cols_y (list, optional): A list of position columns which contains positions relevant to y/vertical-axis. Defaults to ['word_top', 'word_bottom', 'line_top', 'line_bottom', 'block_top', 'block_bottom'].
    
    Returns:
        pandas.DataFrame: A dataframe for XML object with position info to the original starting position.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> xml_word = parse_xmls(xmls, output='word')
        >>> shifted_xml_word = shift_to_standard(xml_word, xmls)
        >>> org_xml_word = shift_to_original(shifted_xml_word, xmls)
    """ 
    l, t, r, b = get_element(xmls, 'pos').split(',')
    l = float(l) * ratio
    t = float(t) * ratio

    start = scaled_xml_word[cols]
    xml_word1 = scaled_xml_word.loc[:, cols_x] + l
    xml_word2 = scaled_xml_word.loc[:, cols_y] + t

    xml_word = pd.concat([start, xml_word1, xml_word2], axis=1, sort=False)
    xml_word = xml_word[scaled_xml_word.columns.to_list()]

    return xml_word


def find_blocks(xmls, start, end):
    """Extract all blocks between two given words.
    
    Args:
        xmls (list): A list of strings containing XML object.
        start (str): The first occurance of this sting marks the start of the extraction.
        end (str): The first occurance of this sting marks the end of the extraction.

    Returns:
        tuple: A tuple containing a list of <Block> from OCR XML which falls between the start and end words, and the value of the pos attribute of the <Page> element.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> for xml in xmls:
        >>>     blocks, page_pos = find_blocks(xml, start, end)
    """ 
    xtree = et.ElementTree(et.fromstring(xmls, parser=etree.XMLParser(recover=True)))
    xroot = xtree.getroot()
    
    blocks = []
    include = False
    for block in xroot.iter('Block'):
        for line in list(block):
            for word in list(line):
                if str(word.tag) == 'W' and str(word.attrib['v']) == start:
                    include = True
                if str(word.tag) == 'W' and str(word.attrib['v']) in end:
                    include = False

        if include:
            blocks.append(dict(block.attrib))
    
    l, t, r, b = get_element(xmls, 'pos').split(',')
    pos = str(l) + ',' + str(t) + ',' + str(r) + ',' + str(b)

    return blocks, pos


def get_block_pos(blocks, page_pos):
    """Vertically reposition a <Block> from OCR data at the top of the page.
    
    Args:
        blocks (list): A list of strings containing <Block> from OCR XML object.
        page_pos (str): The value of the pos attribute of the <Page> element.

    Returns:
        str: A repositioned <Block> at the top of the page.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> for xml in xmls:
        >>>     blocks, page_pos = find_blocks(xml, start, end)
        >>>     if len(blocks) > 0:
        >>>         pos = get_block_pos(blocks, page_pos)
    """ 
    t = []
    b = []

    for block in blocks:
        l1, t1, r1, b1 = block['pos'].split(',')
        t.append(int(t1))
        b.append(int(b1))

    t = min(t)
    b = max(b)

    lp, tp, rp, bp = page_pos.split(',')

    pos = str(lp) + ',' + str(t) + ',' + str(rp) + ',' + str(b)

    return pos


def create_sub_blocks(xmls, blocks, pos):
    """Create XML object from extarted blocks.
    
    Args:
        xmls (list): A list of strings containing XML object.
        blocks (list): A list of strings containing <Block> from OCR XML object.
        pos (str): A repositioned <Block> at the top of the page created using get_block_pos.

    Returns:
        str: A string containing the XML object ready to be persisted.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> for xml in xmls:
        >>>     blocks, page_pos = find_blocks(xml, start, end)
        >>>     if len(blocks) > 0:
        >>>         pos = get_block_pos(blocks, page_pos)
        >>>         new_xml = create_sub_blocks(xml, blocks, pos)
    """ 
    a = json.loads(json.dumps(xmltodict.parse(xmls)))
    
    temps = {}
    temp = []
    for i in range(len(a['Page']['Block'])):
        for j in range(len(blocks)):
            if a['Page']['Block'][i]['@pos'] == blocks[j]['pos']:
                temp.append(a['Page']['Block'][i])
                temps['Block'] = temp

    b = {}
    b['Page'] = temps

    new_xml = xmltodict.unparse(b, pretty=True, full_document=False)
    new_xml = new_xml.replace('\t', '    ').replace('<Page>', '<?xml version="1.0" encoding="utf-16"?>\n<Page xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" pos="' + pos + '" lang="English" printArea="' + pos + '" xdpi="300" ydpi="300">')

    return new_xml


def extract_section(xmls, start, end):
    """Create XML object comprising all blocks between two given words.
    
    Args:
        xmls (list): A list of strings containing XML object.
        start (str): The first occurance of this sting marks the start of the extraction.
        end (str): The first occurance of this sting marks the end of the extraction.

    Returns:
        str: A string containing the XML object comprising all blocks between two given words and is ready to be persisted.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> new_xml = extract_section(xmls, start='3.4', end=('3.5', '3.6', '3.7', '3.8'))
    """ 
    for xml in xmls:
        blocks, page_pos = find_blocks(xml, start, end)
        if len(blocks) > 0:
            pos = get_block_pos(blocks, page_pos)
            return create_sub_blocks(xml, blocks, pos)

    return None


def read_xml(xml_file, page=1, output='line', smart_cell=True):
    """Read a single XML file to DataFrame containing either words or lines.
    
    Args:
        xml_file (str): Fully qualified filename with path.
        page (float, optional): Page number to convert to dataframe. Defaults to 1.
        output (str, optional): The resultant dataframe contains lines if 'line' or 'l', otherwise words. Defaults to 'line'.
        smart_cell (bool, optional): To keep tabular data same as block, place <Table> data in the hierarchy of Table, Row, Cell, Word if True, otherwise Table, Cell, Line, Word. Defaults to True.

    Returns:
        pandas.DataFrame: A dataframe containing the word or line content.

    Examples:
        >>> xml_df = read_xml('../test_data/big_section_34.xml')
    """ 
    with open(xml_file, encoding='utf-16') as f:
        xml_string = f.read()
        f.close()

    return parse_xml(xml_string, page, output, smart_cell)


def parse_xmls(xmls, output='line', smart_cell=True):
    """Convert a list of XMLs to DataFrame containing either words or lines. If only one xml file has been sent, page number defaults to 1, for multi-page files, all pages are put into a single dataframe with page number.
    
    Args:
        xmls (list): A list of strings containing XML object.
        output (str, optional): The resultant dataframe contains lines if 'line' or 'l', otherwise words. Defaults to 'line'.
        smart_cell (bool, optional): To keep tabular data same as block, place <Table> data in the hierarchy of Table, Row, Cell, Word if True, otherwise Table, Cell, Line, Word. Defaults to True.

    Returns:
        pandas.DataFrame: A dataframe containing the word or line content.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
        >>> xml_word = parse_xmls(xmls, output='line')
    """
    all_page_xml = []
    for index, xml in enumerate(xmls):
        all_page_xml.append(parse_xml(xml, index + 1, output, smart_cell))

    all_page_xml = pd.concat(all_page_xml, ignore_index=True)
    if len(all_page_xml) > 0:
        all_page_xml.sort_values(by=['page', 'line_top'], inplace=True)
        all_page_xml.reset_index(drop=True, inplace=True)
    else:
        if output == 'line' or output == 'l':
            df_col = ('block_type', 'table_pos', 'block_left', 'block_top', 'block_right', 'block_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'word', 'page', 'non_ascii_ratio')
        else:
            df_col = ('block_type', 'table_pos', 'block_left', 'block_top', 'block_right', 'block_bottom', 'line_left', 'line_top', 'line_right', 'line_bottom', 'word_left', 'word_top', 'word_right', 'word_bottom', 'word', 'page', 'non_ascii_ratio')
        all_page_xml = pd.DataFrame(columns=df_col)

    return all_page_xml


def read_xml_from_directory(directory, encoding='utf-16'):
    """Read the directory of all xml files.
    
    Args:
        directory (str): Fully qualified foldername with path.
        encoding (str, optional): Text encoding of the XML files. Defaults to 'utf-16'.
        
    Returns:
        list: A list of strings containing XML objects.

    Examples:
        >>> xmls = read_xml_from_directory('../test_data/')
    """ 
    xml_files = []
    for file in os.scandir(directory):
        if file.name.endswith('.xml'):
            xml_files.append(file.name)

    try:
        if xml_files[0].rfind('-') > xml_files[0].rfind('_'):
            xml_files.sort(key=lambda f: int(f[f.rfind('-') + 1:f.rfind('.xml')]))
        else:
            xml_files.sort(key=lambda f: int(re.sub(r'\D', '', f[f.rfind('_') + 1:f.rfind('.xml')])))
    except:
        logger.info('File name is not following the Datacap OCR XML file naming convention. Page sort may not work!')

    xmls = []
    for filename in xml_files:
        with open(directory + '/' + filename, encoding=encoding) as f:
            xml_string = f.read()
            f.close()
        xmls.append(xml_string)
     
    return xmls


# Main function of read the directory of all xml file, combine all the pages into a single dataframe 
def read_directory(directory, output='line', smart_cell=True):
    """Read the directory of all xml files.
    
    Args:
        directory (str): Fully qualified foldername with path.
        output (str, optional): The resultant dataframe contains lines if 'line' or 'l', otherwise words. Defaults to 'line'.
        smart_cell (bool, optional): To keep tabular data same as block, place <Table> data in the hierarchy of Table, Row, Cell, Word if True, otherwise Table, Cell, Line, Word. Defaults to True.
        
    Returns:
        pandas.DataFrame: A dataframe containing the word or line content.

    Examples:
        >>> xml_df = read_directory('../test_data/')
    """ 
    return pd.DataFrame(parse_xmls(read_xml_from_directory(directory), output, smart_cell))
