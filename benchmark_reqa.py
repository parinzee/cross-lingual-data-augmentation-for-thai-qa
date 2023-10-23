import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from utils import encode_in_batch, list_to_ordered_set

import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from collections import defaultdict, Counter
from itertools import combinations

def evaluate(aq_id, question_id, question_all, context_id, context_all, mrr_rank=10, status=True):
    top_1 = 0
    top_5 = 0
    top_10 = 0
    mrr_score = 0
    context_id = np.array(context_id)
    sim_score = np.inner(question_all, context_all)

    aq_id_to_score = defaultdict(list)
    aq_id_to_simscore = defaultdict(list)

    if status == True:
        status_bar = enumerate(tqdm(zip(sim_score, aq_id)))
    else:
        status_bar = enumerate(zip(sim_score, aq_id))

    for idx, (sim, aq) in status_bar:
        index = np.argsort(sim)[::-1]
        index_edit = [context_id[x] for x in index]
        try:
            idx_search = list_to_ordered_set(index_edit).index(question_id[idx])
        except:
            # print(f"Error on Question ID: {question_id[idx]}")
            continue
        
        aq_id_to_score[aq].append(idx_search)
        try:
            aq_id_to_simscore[aq].append(sim[idx_search])
        except IndexError:
            aq_id_to_simscore[aq].append(0)
    
    for aq, idx_searches in aq_id_to_score.items():
        # Get the most common index if possible
        counter = Counter(idx_searches)

        # Get the most common item(s) using most_common()
        most_common_item = counter.most_common(1)
        most_common_count = most_common_item[0][1]

        # Find all items that have the same highest count as the most common item
        most_common_items = [item for item, count in counter.items() if count == most_common_count]
        if len(most_common_items) == 1:
            idx_search = most_common_items[0]
        else:
            # If there are multiple most common items, get the one with the highest similarity score
            # Get the similarity scores
            sim_scores = aq_id_to_simscore[aq]
            # Get the indices of the most common items
            most_common_indices = [idx_searches.index(item) for item in most_common_items]
            # Get the similarity scores of the most common items
            most_common_sim_scores = [sim_scores[idx] for idx in most_common_indices]
            # Get the index of the most common item with the highest similarity score
            idx_search = most_common_indices[np.argmax(most_common_sim_scores)]

        if idx_search == 0:
            top_1 += 1
            top_5 += 1
            top_10 += 1
        elif idx_search < 5:
            top_5 += 1
            top_10 += 1
            
            # # Debugging
            # print(idx_search)
            # print(index_edit)
            # print(question_id[idx])
            # print(f"Question: {questions_data[questions_data['context_id'] == question_id[idx]]['question'].values[0]}")
            # print(f"Context: {context_data[context_data['id'] == question_id[idx]]['context'].values[0]}")
            # print(f"GT Retrieved Context: {context_data[context_data['id'] == index_edit[idx_search]]['context'].values[0]}")
            # # Print all retrieved contexts top-15
            # print("Retrieved Contexts:")
            # for i in range(15):
            #     print(f"{i}: {context_data[context_data['id'] == index_edit[i]]['context'].values[0]}")

            # print("--")

            # if top_5 > 20:
            #     raise Exception("Debugging")
        elif idx_search < 10:
            top_10 += 1
        if idx_search < mrr_rank:
            mrr_score += 1 / (idx_search + 1)
    return (
        top_1 / len(question_all),
        top_5 / len(question_all),
        top_10 / len(question_all),
        mrr_score / len(question_all),
    )


def get_ds(question_data, context_data, aug_question_col=None, aug_context_col=None, aug_ratio=0., context_col="context", questions_col="question", filter_zero_dist=True):
    """Function to return the questions and contexts, and their ids for use in benchmarking"""

    # Assertions to prevent race conditions
    assert aug_question_col is None or aug_context_col is None, "Both aug_question_col and aug_context_col cannot be on at the same time"
    assert aug_question_col is None or aug_question_col in question_data.columns, f"{aug_question_col} not in question_data"
    assert aug_context_col is None or aug_context_col in context_data.columns, f"{aug_context_col} not in context_data"
    if aug_question_col is not None:
        assert aug_question_col.replace("th_", "dis_") in question_data.columns, f"{aug_question_col.replace('th_', 'dis_')} not in question_data"
    if aug_context_col is not None:
        assert aug_context_col.replace("th_", "dis_") in context_data.columns, f"{aug_context_col.replace('th_', 'dis_')} not in context_data"

    # Get the questions and contexts
    questions = question_data[questions_col].values
    # TODO: Refactor this later with actual arguments
    question_ids = question_data["context_id"].values
    # question_ids = question_data["id"].values
    actual_question_ids = question_data["id"].values

    # Get the contexts
    context = context_data[context_col].values
    context_ids = context_data["id"].values

    # Get the augmented contexts
    if aug_context_col:
        aug_context = context_data[aug_context_col].values
        distances = context_data[aug_context_col.replace("th_", "dis_")].values

        if filter_zero_dist:
            aug_context = aug_context[distances > 0]
            aug_context_ids = context_data["id"].values[distances > 0]

            # Filter out the zero distances
            distances = distances[distances > 0]
        else:
            aug_context_ids = context_data["id"].values
        
        # Sample bottom augmented contexts by aug_context_ratio (closest distance)
        if aug_ratio > 0:
            # Get the bottom indices
            bottom_indices = np.argsort(distances)[:int(len(distances) * aug_ratio)]
            # Get the bottom contexts
            aug_context = aug_context[bottom_indices]
            aug_context_ids = aug_context_ids[bottom_indices]
        else:
            # Raise error if aug_context_ratio is 0 but aug_context_col is not None
            assert aug_ratio != 0, "aug_context_ratio cannot be 0 if aug_context_col is not None"
        
        # Augment into context
        context = np.concatenate([context, aug_context], axis=0)
        context_ids = np.concatenate([context_ids, aug_context_ids], axis=0)
    
    elif aug_question_col:
        aug_question = question_data[aug_question_col].values
        distances = question_data[aug_question_col.replace("th_", "dis_")].values

        if filter_zero_dist:
            aug_question = aug_question[distances > 0]
            aug_question_ids = question_data["context_id"].values[distances > 0]
            actual_aug_question_ids = question_data["id"].values[distances > 0]

            # Filter out the zero distances
            distances = distances[distances > 0]
        else:
            aug_question_ids = question_data["context_id"].values
            actual_aug_question_ids = question_data["id"].values
        
        # Sample bottom augmented contexts by aug_context_ratio (closest distance)
        if aug_ratio > 0:
            # Get the bottom indices
            bottom_indices = np.argsort(distances)[:int(len(distances) * aug_ratio)]
            # Get the bottom contexts
            aug_question = aug_question[bottom_indices]
            aug_question_ids = aug_question_ids[bottom_indices]
            actual_aug_question_ids = actual_aug_question_ids[bottom_indices]
        else:
            # Raise error if aug_context_ratio is 0 but aug_context_col is not None
            assert aug_ratio != 0, "aug_context_ratio cannot be 0 if aug_context_col is not None"

        # Augment into context
        questions = np.concatenate([questions, aug_question], axis=0)
        question_ids = np.concatenate([question_ids, aug_question_ids], axis=0)
        actual_question_ids = np.concatenate([actual_question_ids, actual_aug_question_ids], axis=0)
    
    return actual_question_ids, question_ids, context_ids, questions, context


def benchmark_single(aq_ids, q_ids, c_ids, q, c, model_embed):
    # Check if column has a distance counterpart
    question_all = encode_in_batch(model_embed, q, progress=False)
    context_all = encode_in_batch(model_embed, c, progress=False)

    question_ids = q_ids
    context_ids = c_ids

    top_1, top_5, top_10, mrr_score = evaluate(
        aq_ids, question_ids, question_all, context_ids, context_all, status=False
    )
    return top_1, top_5, top_10, mrr_score

if __name__ == "__main__":
    import hashlib
    question_data = pd.read_parquet('questions/data/06_calculate_distance.parquet')
    context_data = pd.read_parquet('contexts/data/06_calculate_distance.parquet')

    question_data["context_id"] = question_data["context"].apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())

    results = []
    model_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    all_augment_cols = [col for col in question_data.columns if col.startswith('th_')]
    unique_sources = list(question_data["source"].unique())
    unique_sources = ["xquad"]

    # First test out on no augmentation
    # for source in tqdm(unique_sources, desc="Iterating through datasets"):
    #     actual_question_ids, question_ids, context_ids, questions, context = get_ds(question_data[question_data["source"] == source], context_data[context_data["source"] == source], aug_question_col=None, aug_ratio=0)
        
    #     top_1, top_5, top_10, mrr_score = benchmark_single(actual_question_ids, question_ids, context_ids, questions, context, model_embed)
    #     print(f"Source: {source}, Augment Col: None, Augment Ratio: 0, Top 1: {top_1}, Top 5: {top_5}, Top 10: {top_10}, MRR Score: {mrr_score}, Number of Augmentations: 0")
    #     results.append({
    #         "source": source,
    #         "augment_col": None,
    #         "augment_ratio": 0,
    #         "top_1": top_1,
    #         "top_5": top_5,
    #         "top_10": top_10,
    #         "mrr_score": mrr_score,
    #         "num_augmentations": 0
    #     })
    
    # Test single augmentation
    # for source in tqdm(unique_sources, desc="Iterating through datasets"):
    #     for col in tqdm(all_augment_cols, desc="Iterating through columns", leave=False):
    #         for ratio in tqdm(range(1, 11), desc="Iterating through ratios", leave=False, total=10, miniters=0):
    #             ratio = ratio / 10
    #             actual_question_ids, question_ids, context_ids, questions, context = get_ds(question_data[question_data["source"] == source], context_data[context_data["source"] == source], aug_question_col=col, aug_ratio=ratio)
                
    #             top_1, top_5, top_10, mrr_score = benchmark_single(actual_question_ids, question_ids, context_ids, questions, context, model_embed)
    #             print(f"Source: {source}, Augment Col: {col}, Augment Ratio: {ratio}, Top 1: {top_1}, Top 5: {top_5}, Top 10: {top_10}, MRR Score: {mrr_score}, Number of Augmentations: 1")
    #             results.append({
    #                 "source": source,
    #                 "augment_col": [col],
    #                 "augment_ratio": ratio,
    #                 "top_1": top_1,
    #                 "top_5": top_5,
    #                 "top_10": top_10,
    #                 "mrr_score": mrr_score,
    #                 "num_augmentations": 1
    #             })


    for num_to_augment in tqdm(range(5, len(all_augment_cols) + 1), desc="Testing number of augmentations"):
        for augment_cols in tqdm(list(combinations(all_augment_cols, num_to_augment)), desc="Iterating through combinations", leave=False):
            for source in tqdm(unique_sources, desc="Iterating through datasets"):
                    for ratio in tqdm(range(1, 11), desc="Iterating through ratios", leave=False, total=10, miniters=0):
                        ratio = ratio / 10

                        # Get the data
                        actual_question_ids, question_ids, context_ids, questions, context = get_ds(question_data[question_data["source"] == source], context_data[context_data["source"] == source], aug_question_col=augment_cols[0], aug_ratio=ratio)

                        for col in augment_cols[1:]:
                            temp_actual_question_ids, temp_question_ids, temp_context_ids, temp_questions, temp_context = get_ds(question_data[question_data["source"] == source], context_data[context_data["source"] == source], aug_question_col=col, aug_ratio=ratio)
                            actual_question_ids = np.concatenate([actual_question_ids, temp_actual_question_ids], axis=0)
                            question_ids = np.concatenate([question_ids, temp_question_ids], axis=0)
                            context_ids = np.concatenate([context_ids, temp_context_ids], axis=0)
                            questions = np.concatenate([questions, temp_questions], axis=0)
                            context = np.concatenate([context, temp_context], axis=0)
                        
                        top_1, top_5, top_10, mrr_score = benchmark_single(actual_question_ids, question_ids, context_ids, questions, context, model_embed)
                        print(f"Source: {source}, Augment Col: {augment_cols}, Augment Ratio: {ratio}, Top 1: {top_1}, Top 5: {top_5}, Top 10: {top_10}, MRR Score: {mrr_score}, Number of Augmentations: {num_to_augment}")

                        results.append({
                            "source": source,
                            "augment_cols": augment_cols,
                            "augment_ratio": ratio,
                            "top_1": top_1,
                            "top_5": top_5,
                            "top_10": top_10,
                            "mrr_score": mrr_score,
                            "num_augmentations": num_to_augment
                        })

    results = pd.DataFrame(results)
    print(results)
    results.to_csv("benchmark_reqa_results.csv", index=False)