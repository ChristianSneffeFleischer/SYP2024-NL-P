### This file contains functions which are used to load and format data to use in NER models
### Is not supposed to be used as a script 

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer

def read_jsonl(filepath, start_id, end_id):
    '''Read JSON Lines (.jsonl) files. 
    Final format is list[dict{"id": int, "text": str, "label": list[list], "Comments": list}]'''

    # Read data
    data = []
    with open(filepath) as infile:
        for line in infile:
            data.append(json.loads(line))

    # Fix id's
    id_range = iter(list(range(start_id, end_id + 1)))
    for entry in data:
        entry['id'] = next(id_range)
    
    return data

def continuous_string(data):
    '''Collapse data file into single string and label list.'''

    output_string = ''
    output_labels = []
    for i, article in enumerate(data):

        # Append label data to list
        for start_idx, end_idx, label in article['label']:
            output_labels.append((start_idx + len(output_string), end_idx + len(output_string), label))

        # Append article text to string
        output_string += article['text']
        if i != len(data) - 1:
            output_string += ' ' # Add space between articles

    return output_string, output_labels

# def split_data(data):
#     '''Split input data dictionary into text and annotations lists.'''

#     # Define lists to store data
#     texts = []
#     annotations = []

#     for article in data:
#         texts.append(article['text'])
#         annotations.append([(start_idx, end_idx, label) for start_idx, end_idx, label in article['label']])

#     return texts, annotations

def label_distribution(annotations):
    '''
    Calculate the distribution of labels within a given set of annotations.

    Each annotation is expected to be a tuple with structure (_, _, label),
    where 'label' is the entity type or category.

    Args:
        annotations (list of tuples): A list of annotations, where each 
                                      annotation is represented as a tuple.

    Returns:
        dict: A dictionary where keys are the labels (entity types) and values
              are the counts of each label in the input annotations.
    '''
    counts = {}
    for _, _, label in annotations:
        counts[label] = counts.get(label, 0) + 1
    return counts

