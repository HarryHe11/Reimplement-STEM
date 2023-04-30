import csv
import os
import pickle
from collections import Counter
from typing import List, Any, Dict, Tuple, Set, Iterable, Sequence
from operator import itemgetter
from itertools import combinations, starmap, groupby, product, chain, islice, takewhile

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

from tqdm import tqdm

import matplotlib.pyplot as plt
from CreateDebate_Conversation_Builder import DataFrameConversationReader

from conversant.Conversation import Conversation, get_subtree_messages, Message

from conversant.InteractionsGraph import InteractionsGraph
from conversant.InteractionsGraph import PairInteractionsData

from classifiers.base_stance_classifier import BaseStanceClassifier
from classifiers.greedy_stance_classifier import MSTStanceClassifier
from classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
# from stance_classification.draw_utils import new_figure

def get_majority_vote(labels: List[int]) -> int:
    return int(np.mean(labels) >= 0.5)

def get_author_labels(c: Conversation) -> Dict[Any, int]:
    authors_post_labels = {}
    for depth, node in c.iter_conversation():
        label = node.message.discussion_stance_id
        author = node.message.author
        current_author_labels = authors_post_labels.setdefault(author, [])
        current_author_labels.append(label)

    result_labels = {a: get_majority_vote(labels) for a, labels in authors_post_labels.items()}
    return result_labels

def get_ordered_candidates_for_pivot(graph: nx.Graph, weight_field: str = "weight") -> Sequence[Any]:
    inv_weight_field = "inv_weight"
    for _, _, pair_data in graph.edges(data=True):
        weight = pair_data[weight_field]
        pair_data[inv_weight_field] = 1 / weight

    node_centralities = nx.closeness_centrality(graph, distance=inv_weight_field)
    return list(map(itemgetter(0), sorted(node_centralities.items(), key=itemgetter(1), reverse=True)))

def get_pivot_node(graph: nx.Graph, labeled_authors: Set[Any], weight_field: str = "weight") -> Any:
    candidates = get_ordered_candidates_for_pivot(graph, weight_field=weight_field)
    return next(iter(filter(labeled_authors.__contains__, candidates)), None)

def extend_preds(graph: nx.Graph, seed_node: Any, core_authors_preds: Dict[Any, int]) -> Dict[Any, int]:
    extended_results = dict(core_authors_preds.items())
    for (n1, n2) in nx.bfs_edges(graph, source=seed_node):
        if n2 not in extended_results:
            n1_label = extended_results[n1]
            extended_results[n2] = 1 - n1_label

    return extended_results

def get_authors_labels_in_conv(conv: Conversation, author_labels_per_conversation) -> Dict[Any, int]:
    if conv.id not in author_labels_per_conversation:
        return None

    return author_labels_per_conversation[conv.id]

def get_author_preds(clf: BaseStanceClassifier, pivot: Any, authors_labels) -> Dict[Any, int]:
    support_label = authors_labels[pivot]
    opposer_label = 1 - support_label
    supporters = clf.get_supporters()
    opposers = clf.get_complement()
    preds = {}
    for supporter in supporters:
        preds[supporter] = support_label
    for opposer in opposers:
        preds[opposer] = opposer_label

    return preds

def get_maxcut_results(graph: InteractionsGraph, op: Any) -> MaxcutStanceClassifier:
    maxcut = MaxcutStanceClassifier(weight_field=graph.WEIGHT_FIELD)
    maxcut.set_input(graph.graph, op)
    maxcut.classify_stance()
    return maxcut

def get_greedy_results(graph: InteractionsGraph, op: Any) -> BaseStanceClassifier:
    clf = MSTStanceClassifier()#weight_field=graph.WEIGHT_FIELD)
    clf.set_input(graph.graph)
    clf.classify_stance(op)
    return clf

def align_gs_with_predictions(authors_labels: Dict[Any, int], author_preds: Dict[Any, int]) -> Tuple[List[int], List[int]]:
    y_true, y_pred = [], []
    for author, true_label in authors_labels.items():
        pred = author_preds.get(author, None)
        if pred is None: continue

        y_true.append(true_label)
        y_pred.append(pred)

    return y_true, y_pred

def predict_for_partition(true: List[int], preds: List[int]) -> Tuple[List[int], List[int]]:
    acc = accuracy_score(true, preds)
    if acc < 0.5:
        preds = [1-l for l in preds]

    return true, preds

def get_best_preds(true_labels: Dict[Any, int], pred_labels: Dict[Any, int]) -> Dict[Any, int]:
    true, preds = align_gs_with_predictions(true_labels, pred_labels)
    acc = accuracy_score(true, preds)
    if acc < 0.5:
        return {k: (1-  l) for k, l in pred_labels.items()}

    return pred_labels

def get_posts_preds(conv: Conversation, post_labels: Dict[Any, int], author_preds: Dict[Any, int]) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    posts_true, posts_pred = {}, {}
    conv_id = conv.id
    for depth, node in conv.iter_conversation():
        label = post_labels.get((conv_id, node.message.node_id), None)
        if label is None: continue
        pred = author_preds.get(node.message.author, None)
        if pred is None: continue

        posts_true[node.message.node_id] = label
        posts_pred[node.message.node_id] = pred

    return posts_true, posts_pred