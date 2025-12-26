# Coordinate Matrix Machine (CM2)

A human-level concept learning approach for document classification, using Sci-kit Learn implementation in Python.

The advantages of our algorithm are:
1. Better accuracy
2. Faster computation
3. Generic, expandable, and extendable
4. Explainable
5. Robust against unbalanced classes
6. Does not require a large number of labelled data

---

## Prerequisites

### 1. Python Packages

The following Python packages must be installed in your environment:

- Pandas
- Numpy
- Sci-kit Learn
- Statistics

##  Project Structure

```
│
├── data/                           # Test data
│
├── lib/                            # All internal codes.
│   ├── __init__.py                 # Empty file.
│   ├── classifier.py               # Classifier definition.
│   ├── cm2.py                      # Coordinate Matrix Machine implementation.
│   ├── ienum.py                    # Indexable enumeration.
│   ├── model.py                    # Base classes for different model types.
│   └── xml_reader.py               # Read and process XML files.
│
├── Training Data Sample/           # A sample of how training data needs to be annotated.
│   ├── Commonwealth Bank/          # Class label - level 1.
│       ├── CBA_CC_E_Mail_Platinum/ # Class label - level 2.
│           ├── sample.xml          # XML file received from the OCR engine.
│           ├── reference.csv       # CSV file with the annotation of the features.
│
├── Encode_Training_Data.ipynb      # Convert XML Files into a Pandas data frame to be used with CM2.
├── Pipeline_Create.ipynb           # Code to train a model and create model.pkl.
├── Pipeline_Test.ipynb             # Code to test the model on test data.
├── model.pkl                       # Saved model created using Pipeline_Create notebook.
├── trained_data.pickle             # Encoded training created using Encode_Training_Data notebook.
└── README.md                       # This file.
```
## Execution Environment

All codes were run on a MacBook Air M2. 

## CM2 Implementation

1. Run the Encode_Training_Data script pointing to the Training Data folder. Provided one is only a sample showing how to annotate your data. This would create trained_data.pickle with your training data.
2. Run the Pipeline_Create script to fit a CM2 model. This script will save a trained model as model.pkl for your data.
3. Use the Pipeline_Test script to use the predict method of the trained CM2 model on the provided test data.
