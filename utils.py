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

def train_test_split_stratify(data, test_size, seed = None, shuffle = False):
    '''
    Perform a stratified split of the dataset into training and test sets based on 
    the distribution of labels, using KMeans clustering to form stratified groups.

    Args:
        data (list of dicts): The dataset, where each element is a dictionary 
                              that must include a 'label' key among others.
        test_size (float): The proportion of the dataset to include in the test split.
        seed (int, optional): Random seed for reproducibility of results. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the training and test datasets 
                                  after splitting. Defaults to False.

    Returns:
        tuple: A tuple containing two lists, (train_set, test_set), representing 
               the training and test datasets, respectively.
    '''
    # Compute label diversities
    diversities = [label_distribution(article['label']) for article in data]

    # Convert distributions to vectors for clustering
    vec = DictVectorizer(sparse = False)
    diversity_matrix = vec.fit_transform(diversities)

    # Perform clustering
    kmeans = KMeans(n_clusters = 10, random_state = seed, n_init = 10)
    clusters = kmeans.fit_predict(diversity_matrix)

    # Group data by clusters
    clustered_data = {i: [] for i in range(kmeans.n_clusters)}
    for article, cluster in zip(data, cluster):
        clustered_data[cluster].append(article)

    # Stratified sampling from each cluster
    train_set = []
    test_set = []
    for articles in clustered_data.values():
        train, test = train_test_split(articles, test_size = test_size, random_state = seed)
        train_set.extend(train)
        test_set.extend(test)

    if shuffle:
        np.random.shuffle(train_set)
        np.random.shuffle(test_set)

    return train_set, test_set    

    # # Create bins for stratified sampling
    # bins = {tag: [] for tag in labelset}
    # for article in data:
    #     for tag in bins.keys():
    #         if any(label == tag for _, _, label in article['label']):
    #             bins[tag].append(article)

    # # Sample from bins
    # train_set = []
    # test_set = []
    # for articles in bins.values():
    #     train, test = train_test_split(articles, test_size = test_size, random_state = seed)
    #     train_set.extend(train)
    #     test_set.extend(test)

    # return train_set, test_set