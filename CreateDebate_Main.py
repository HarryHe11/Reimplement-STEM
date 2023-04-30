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
from conversant.InteractionsGraph import InteractionsGraph,PairInteractionsData
from CreateDebate_utils import get_pivot_node, get_author_preds,get_author_labels, get_authors_labels_in_conv, get_greedy_results, get_maxcut_results,get_subtree_messages,\
    get_posts_preds,get_best_preds,get_majority_vote,get_ordered_candidates_for_pivot,align_gs_with_predictions, extend_preds
from classifiers.base_stance_classifier import BaseStanceClassifier
from classifiers.greedy_stance_classifier import MSTStanceClassifier
from classifiers.maxcut_stance_classifier import MaxcutStanceClassifier


from Logger import Logger
import sys

type = sys.getfilesystemencoding()
sys.stdout = Logger(r"D:\HKU\MM2023\data\CreateDebates\stance\stem_createdebate.txt")

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")



pasre_strategy = {
    "node_id": "post_id",
    "author": "author_id",
    "timestamp": "creation_date",
    "parent_id": "parent_post_id",
    "topic": "topic",
    "discussion_id": "discussion_id",
    "text": "text",
    "discussion_stance_id": "discussion_stance_id",
    "rebuttal": "discussion_stance_id",
    "thread_id":"thread_id",
    "branch":"branch",
    }

df = pd.read_csv(r'D:\HKU\MM2023\data\CreateDebates\stance\branch_create_debate.csv')

sub_convs = []
parser = DataFrameConversationReader(pasre_strategy)
for thread_id in df['thread_id'].unique():
    temp_df = df.loc[df['thread_id']==thread_id]
    conv = parser.parse(temp_df)
    sub_convs.append(conv)


post_labels={}
for conv in sub_convs:
    # print("participants", len(conv.participants))
    # print("conv_root", conv.root.message.node_id)
    for _,node in conv.iter_conversation():
        # print((conv.id, node.message.node_id))
        post_labels[(conv.id, node.message.node_id)] = node.message.discussion_stance_id

print("all label:", len(list(post_labels.values())))
author_labels_per_conversation = {c.id: get_author_labels(c) for c in sub_convs}
author_labels_per_conversation = {k: v for k, v in author_labels_per_conversation.items() if len(v) > 0 and not (len(v) == 1 and None in v)}
# print('author_labels_per_conversation',len(author_labels_per_conversation))
# print(sum(len(v) for v in author_labels_per_conversation.values()))



interaction_graph = InteractionsGraph()

convs_by_id: Dict[Any, Conversation] = {}
full_graphs: Dict[Any, InteractionsGraph] = {}
core_graphs: Dict[Any, InteractionsGraph] = {}
maxcut_results: Dict[Any, MaxcutStanceClassifier] = {}
pivot_nodes = {}

author_predictions: Dict[Any, Dict[str, Dict[Any, int]]] = {}
posts_predictions: Dict[Any, Dict[str, Dict[Any, int]]] = {}



empty_core = []
unlabeled_conversations = []
unlabeled_op = []
insufficient_author_labels = []
too_small_cut_value = []
op_not_in_core = []
large_graphs = []
single_author_conv = []

extend_results = False
naive_results = False

def calc_weight(interactions: PairInteractionsData) -> float:
    n_replies = interactions["replies"]
    # n_quotes = interactions["quotes"]
    return n_replies
    # return n_quotes

count_conv = 0
for i, conv in tqdm(enumerate(sub_convs)):

    count_conv += 1
    authors_labels = get_authors_labels_in_conv(conv, author_labels_per_conversation)
    if authors_labels is None:
        unlabeled_conversations.append(i)
        continue

    if len(authors_labels) == 0:
        insufficient_author_labels.append(i)
        continue

    interaction_graph = InteractionsGraph()
    interaction_graph.parse_from_conv(conv)
    zero_edges = [(v, u) for v, u, d in interaction_graph.graph.edges(data=True) if d["weight"] == 0]
    interaction_graph.graph.remove_edges_from(zero_edges)

    if len(conv.participants) <= 1:
        single_author_conv.append(i)
        continue

    convs_by_id[conv.id] = conv
    full_graphs[conv.id] = interaction_graph

    pivot_node = get_pivot_node(interaction_graph.graph, authors_labels, weight_field="weight")
    pivot_nodes[conv.id] = pivot_node

    mst = get_greedy_results(interaction_graph, pivot_node)
    preds = get_author_preds(mst, pivot_node, authors_labels)
    author_predictions[conv.id] = {"mst": preds}


    core_interactions = interaction_graph.get_core_interactions()
    core_graphs[conv.id] = core_interactions
    if core_interactions.graph.size() == 0:
        empty_core.append(i)
        continue

    components = list(nx.connected_components(core_interactions.graph))
    core_interactions = core_interactions.get_subgraph(components[0])
    pivot_node = get_pivot_node(core_interactions.graph, authors_labels, weight_field="weight")
    maxcut = get_maxcut_results(core_interactions, pivot_node)
    if maxcut.cut_value < 3:
        too_small_cut_value.append(i)
        continue

    maxcut_results[conv.id] = maxcut

    preds = get_author_preds(maxcut, pivot_node, authors_labels)
    author_predictions[conv.id]["core"] = preds

    preds = extend_preds(interaction_graph.graph, pivot_node, preds)
    author_predictions[conv.id]["full"] = preds


print(f"total number of conversations (in all topics): {len(sub_convs)}")
print(f"total number of conversations (in the relevant topics): {count_conv}")
print(f"total number of conversations with labeled authors (in all topics): {len(author_labels_per_conversation)}")
print(f"total number of conversations with labeled authors (in the relevant topics): {count_conv - len(unlabeled_conversations)}")

print(f"number of conversations in eval: {len(convs_by_id)}")
print(f"number of conversations with core in eval: {len(core_graphs)}")
all_authors_in_eval = set(chain(*[predictions["mst"].keys() for predictions in author_predictions.values()]))
print(f"number of unique authors in eval: {len(all_authors_in_eval)}")
all_authors_in_core_eval = set(chain(*[predictions.get("core", {}).keys() for predictions in author_predictions.values()]))
print(f"number of unique authors in core: {len(all_authors_in_core_eval)}")

labeled_authors = sum(len(v) for v in author_labels_per_conversation.values())
print(f"total number of labeled authors: {labeled_authors}")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(f"number of conversations with single author: {len(single_author_conv)}")
print(f"number of conversations with empty core: {len(empty_core)}")
print(f"number of conversations with op not in core: {len(op_not_in_core)}")
print(f"number of conversations with too large core: {len(large_graphs)}")
print(f"number of conversations with too small cut value: {len(too_small_cut_value)}")
print(f"number of unlabeled conversations: {len(unlabeled_conversations)}")
print(f"number of conversations with unlabeled op: {len(unlabeled_op)}")
print(f"number of conversations with insufficient labeled authors: {len(insufficient_author_labels)}")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# topics = ['abortion', 'gayRights', 'obama', 'marijuana']
# for test_topic in topics:
#
#     print("Topic: " +test_topic)
#
#     for predictor in ["core", "full"]:
    #     all_true, all_pred = [], []
    #     all_true_best, all_pred_best = [], []
    #
    #     accuracies = []
    #     best_accuracies = []
    #     for conv_id, predictions in author_predictions.items():
    #         conv = convs_by_id[conv_id]
    #         topic = conv.root.message.topic
    #         if topic != test_topic:
    #             continue
    #         author_labels = get_authors_labels_in_conv(conv, author_labels_per_conversation)
    #         author_preds = predictions.get(predictor, None)
    #         if author_preds is None: continue
    #
    #         y_true, y_pred = align_gs_with_predictions(author_labels, author_preds)
    #         all_true.extend(y_true)
    #         all_pred.extend(y_pred)
    #         accuracies.append(accuracy_score(y_true, y_pred))
    #
    #         best_preds = get_best_preds(author_labels, author_preds)
    #         y_true, y_pred = align_gs_with_predictions(author_labels, best_preds)
    #         all_true_best.extend(y_true)
    #         all_pred_best.extend(y_pred)
    #         best_accuracies.append(accuracy_score(y_true, y_pred))
    #
    #
    #     print(f"Showing results of predictor: {predictor}")
    #     print("acc ---- (macro):", np.mean(accuracies))
    #     print("acc best (macro):", np.mean(best_accuracies))
    #     print("acc ---- (micro):", accuracy_score(all_true, all_pred))
    #     print("acc best (micro):", accuracy_score(all_true_best, all_pred_best))
    #
    #     print(classification_report(all_true, all_pred))
    #     print(f"\n\t\tResults for best partition (regardless for stance assignment")
    #     print(classification_report(all_true_best, all_pred_best))


for predictor in ["core", "full"]:
    print(f"Showing Overall results of predictor: {predictor}")

    print("---------------------------------------------------")
    # print("Topic: " +test_topic)
    all_true, all_pred = [], []
    all_true_best, all_pred_best = [], []
    accuracies = []
    best_accuracies = []
    for conv_id, predictions in author_predictions.items():
        conv = convs_by_id[conv_id]
        topic = conv.root.message.topic
        # if topic != test_topic:
        #     continue
        author_labels = get_authors_labels_in_conv(conv, author_labels_per_conversation)
        author_preds = predictions.get(predictor, None)
        if author_preds is None: continue

        posts_true, posts_preds = get_posts_preds(conv, post_labels, author_preds)

        y_true, y_pred = align_gs_with_predictions(posts_true, posts_preds)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        accuracies.append(accuracy_score(y_true, y_pred))

        best_preds = get_best_preds(posts_true, posts_preds)
        y_true, y_pred = align_gs_with_predictions(posts_true, best_preds)
        all_true_best.extend(y_true)
        all_pred_best.extend(y_pred)
        best_accuracies.append(accuracy_score(y_true, y_pred))

    print("acc ---- (macro):", np.mean(accuracies))
    print("acc best (macro):", np.mean(best_accuracies))
    print("acc ---- (micro):", accuracy_score(all_true, all_pred))
    print("acc best (micro):", accuracy_score(all_true_best, all_pred_best))
    print("---------------------------------------------------")

topics = ['abortion', 'gayRights', 'obama', 'marijuana']


for predictor in ["core", "full"]:
    print(f"Showing results of predictor: {predictor}")
    for test_topic in topics:
        print("---------------------------------------------------")
        print("Topic: " +test_topic)
        all_true, all_pred = [], []
        all_true_best, all_pred_best = [], []
        accuracies = []
        best_accuracies = []
        for conv_id, predictions in author_predictions.items():
            conv = convs_by_id[conv_id]
            topic = conv.root.message.topic
            if topic != test_topic:
                continue
            author_labels = get_authors_labels_in_conv(conv, author_labels_per_conversation)
            author_preds = predictions.get(predictor, None)
            if author_preds is None: continue

            posts_true, posts_preds = get_posts_preds(conv, post_labels, author_preds)

            y_true, y_pred = align_gs_with_predictions(posts_true, posts_preds)
            all_true.extend(y_true)
            all_pred.extend(y_pred)
            accuracies.append(accuracy_score(y_true, y_pred))

            best_preds = get_best_preds(posts_true, posts_preds)
            y_true, y_pred = align_gs_with_predictions(posts_true, best_preds)
            all_true_best.extend(y_true)
            all_pred_best.extend(y_pred)
            best_accuracies.append(accuracy_score(y_true, y_pred))

        # print("acc ---- (macro):", np.mean(accuracies))
        # print("acc best (macro):", np.mean(best_accuracies))
        print("acc ---- (micro):", accuracy_score(all_true, all_pred))
        print("acc best (micro):", accuracy_score(all_true_best, all_pred_best))
        print("---------------------------------------------------")


    # print(classification_report(all_true, all_pred))
    # print(f"\n\tResults for best partition (regardless for stance assignment")
    # print(classification_report(all_true_best, all_pred_best))
