import csv
import os
import pickle
from collections import Counter
from typing import List, Any, Dict, Tuple, Set, Iterable, Sequence
from operator import itemgetter
from itertools import takewhile

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

from tqdm import tqdm

import matplotlib.pyplot as plt

data_path = '.\CreateDebates\stance'


def load_authors_map(authors_root_path: str) -> Dict[str, str]:
    post_authors_map = {}
    for topic_name in os.listdir(authors_root_path):
        topic_dirpath = os.path.join(authors_root_path, topic_name)
        for discussion_file in os.listdir(topic_dirpath):
            discussion_path = os.path.join(topic_dirpath, discussion_file)
            with open(discussion_path, 'r') as f:
                post_author_pairs = list(
                    map(lambda l: tuple(map(str.strip, l.strip().split(' ', 1))), f))
                try:
                    post_full_id_author_pairs = [
                        (f"{topic_name}.{post_id}", author) for post_id, author in post_author_pairs]
                    post_authors_map.update(post_full_id_author_pairs)
                except:
                    print([e for e in post_author_pairs if len(e) != 2])
                    raise

    return post_authors_map


posts_author_path = data_path+"/authors"
authors_map = load_authors_map(posts_author_path)
print(len(authors_map))


def get_author(topic: str, post_id: str) -> str:
    return authors_map.get(post_id, None)


def get_record_from_post(topic_dir: str, meta_file: str) -> Dict[str, Any]:
    topic = topic_dir.split("\\")[-1]
    discussion_id = topic + "." + "".join(takewhile(str.isalpha, meta_file))
    meta_filepath = os.path.join(topic_dir, meta_file)
    text_filepath = os.path.join(
        topic_dir, meta_filepath.split(".")[0] + ".data")
    with open(text_filepath, 'r', encoding='utf-8') as text_f:
        text = text_f.read().strip()

    """ example to meta file content:
        ID=24
        PID=23
        Stance=-1
        rebuttal=oppose
    """
    record = {}
    with open(meta_filepath, 'r') as meta_f:
        post_id = discussion_id + next(meta_f).strip().split("=")[1]
        parent_id_str = next(meta_f).strip().split("=")[1]
        parent_post_id = (
            discussion_id + parent_id_str) if parent_id_str != "-1" else None
        stance_str = next(meta_f).strip().split("=")[1]
        stance = int(bool(int(stance_str) + 1)
                     ) if len(stance_str) > 0 else None
        rebuttal = next(meta_f).strip().split("=")[1]
        author_id = get_author(topic, post_id)
        if parent_post_id == -1:
            parent_post_id = None

        record.update(
            dict(topic=topic, discussion_id=discussion_id, post_id=post_id, author_id=author_id, creation_date=-1,
                 parent_post_id=parent_post_id, text=text, discussion_stance_id=stance, rebuttal=rebuttal)
        )
    return record


data_root_dir = data_path
records = []
for topic_dirname in os.listdir(data_root_dir):
    if topic_dirname == "author":
        continue
    topic_dirpath = os.path.join(data_root_dir, topic_dirname)
    if not os.path.isdir(topic_dirpath):
        continue
    for post_file in os.listdir(topic_dirpath):
        if post_file.endswith(".meta"):
            record = get_record_from_post(topic_dirpath, post_file)
            records.append(record)

print("all records:", len(records))
df = pd.DataFrame.from_records(records)

unified_data_path = data_root_dir+"/all_records.csv"
df.to_csv(unified_data_path, index=None)

# Extract IDs of all top-level posts
thread_ids = df['post_id'].loc[df['parent_post_id'].isnull()].to_list()

# Assign thread ID to top-level posts
df.loc[df['parent_post_id'].isnull(), 'thread_id'] = df['post_id']
Top_ids = []
not_found_ids = []


def build_branch(df, now_id, branch, is_end=True):
    temp_df = df.loc[df['post_id'] == now_id]
    branch.append(now_id)
    if len(temp_df) == 0:  # df里找不到post了
        print("not_found_post_id:", now_id)
        not_found_ids.append(now_id)
        return branch
    if temp_df['thread_id'].iloc[0] == now_id:  # 是top了
        Top_ids.append(now_id)
        return branch
    else:
        next_id = temp_df['parent_post_id'].iloc[0]
        if len(df.loc[df['post_id'] == next_id]) > 0:
            branch = build_branch(df, next_id, branch,
                                  is_end=False)  # next round
        else:
            # print("not_found_parent_id:", next_id)
            not_found_ids.append(next_id)
            return branch
    return branch


def branch_process(df):
    df['branch'] = ''

    for idx in tqdm(range(len(df))):
        branch = []
        now_id = df['post_id'].iloc[idx]
        branch = build_branch(df, now_id, branch)
        branch.reverse()
        df['branch'].iloc[idx] = branch
        df['thread_id'].iloc[idx] = branch[0]


branch_process(df)


# print("Top_ids ", set(Top_ids))
# print(len(list(set(Top_ids))))

# print("not_found_ids ", set(not_found_ids))
# print(len(list(set(not_found_ids))))
df.info()
df.to_csv(data_root_dir+"/branch_create_debate.csv", index=False)
