import warnings
import numpy as np
import pandas as pd
import wandb
from huggingface_hub import login, delete_repo
from tqdm.autonotebook import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split
import ast
import pandas as pd
from datasets import Dataset
import collections
import gc
from datasets import load_metric
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          DefaultDataCollator, Trainer, TrainingArguments)
import argparse
import os
from utils import seed_everything, convert_row_to_simple_transformers_format, merge_qas
import time
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Set to false to prevent deadlock

wandb.login()
# login()

SEED = 42 # Gets overwritten by argparse anyway

dataset = pd.read_parquet("questions/data/06_calculate_distance.parquet")
dataset

# Due to some serialization issues, the dict column must be changed back to a real dictionary instead of a string
dataset["answers"] = dataset["answers"].apply(ast.literal_eval)

# Add None so that the first benchmark only consists of the original questions
all_augment_cols = [None] + [col for col in dataset.columns if col.startswith('th_')]
print("Augment columns:", all_augment_cols)

def get_ds(aug_col=None, aug_ratio=0., return_hf=False, use_slem=False, use_bleu=False, select_cosine_threshold=None):
    # Check that use_slem and use_bleu are not both True
    if use_slem and use_bleu:
        raise ValueError("use_slem and use_bleu cannot both be True.")

    if not return_hf:
        # Filter out test_sets
        test_set = dataset[(dataset["source"] == "xquad") | (dataset["source"] == "tydiqa")].apply(convert_row_to_simple_transformers_format, axis=1)
        train_set, val_set = train_test_split(
            (dataset[~dataset.index.isin(test_set.index)]).apply(convert_row_to_simple_transformers_format, axis=1),
            test_size=0.2,
            random_state=SEED
        )

        if aug_col and aug_ratio == 0.:
            raise ValueError("Specify aug_ratio, otherwise no augmentation will be added.")

        if not aug_col and aug_ratio != 0.:
            raise ValueError("Specify aug_col, otherwise no augmentation will be added.")

        if aug_col and aug_ratio != 0.:
            base_col = "_".join(aug_col.split("_")[1:])

            sorted_ds = dataset.copy()

            if select_cosine_threshold:
                # Filter out rows with cosine distance > select_cosine_threshold
                sorted_ds = sorted_ds[sorted_ds[f"dis_{base_col}"] <= select_cosine_threshold]

            if use_slem:
                sorted_ds = sorted_ds.sort_values(f"slem_{base_col}", ascending=False)
            elif use_bleu:
                sorted_ds = sorted_ds.sort_values(f"bleu_{base_col}", ascending=False)
            else:
                sorted_ds = sorted_ds.sort_values(f"dis_{base_col}")
            sorted_ds = sorted_ds[sorted_ds.index.isin(train_set.index)]
            sorted_ds = sorted_ds.iloc[:round(len(sorted_ds) * aug_ratio)]
            sorted_ds["question"] = sorted_ds[aug_col]
            sorted_ds = sorted_ds.apply(convert_row_to_simple_transformers_format, axis=1)

            train_set = pd.concat([train_set, sorted_ds])

        train_set = merge_qas(list(train_set))
        val_set = merge_qas(list(val_set))
        test_set = merge_qas(list(test_set))

        return train_set, val_set, test_set

    else:
        # Filter out test_sets
        test_set = dataset[(dataset["source"] == "xquad") | (dataset["source"] == "tydiqa")]
        train_set, val_set = train_test_split(
            dataset[~dataset.index.isin(test_set.index)],
            test_size=0.2,
            random_state=SEED
        )

        if aug_col and aug_ratio == 0.:
            raise ValueError("Specify aug_ratio, otherwise no augmentation will be added.")

        if not aug_col and aug_ratio != 0.:
            raise ValueError("Specify aug_col, otherwise no augmentation will be added.")

        if aug_col and aug_ratio != 0.:
            base_col = "_".join(aug_col.split("_")[1:])

            sorted_ds = dataset.copy()

            if select_cosine_threshold:
                # Filter out rows with cosine distance > select_cosine_threshold
                sorted_ds = sorted_ds[sorted_ds[f"dis_{base_col}"] <= select_cosine_threshold]

            if use_slem:
                sorted_ds = sorted_ds.sort_values(f"slem_{base_col}", ascending=False)
            elif use_bleu:
                sorted_ds = sorted_ds.sort_values(f"bleu_{base_col}", ascending=False)
            else:
                sorted_ds = sorted_ds.sort_values(f"dis_{base_col}")
            sorted_ds = sorted_ds[sorted_ds.index.isin(train_set.index)]
            sorted_ds = sorted_ds.iloc[:round(len(sorted_ds) * aug_ratio)]
            sorted_ds["question"] = sorted_ds[aug_col]

            train_set = pd.concat([train_set, sorted_ds])

        return Dataset.from_pandas(train_set), Dataset.from_pandas(val_set), Dataset.from_pandas(test_set)

def get_training_args(exp_name: str, push_to_hub=True, use_slem=False, use_bleu=False, select_cosine_threshold=None):
    # Check that use_slem and use_bleu are not both True
    if use_slem and use_bleu:
        raise ValueError("use_slem and use_bleu cannot both be True.")
    
    output_dir = "models/claq-qa-th-wangchanberta-"

    if use_slem:
        output_dir = output_dir + f"slem-{exp_name}"
    elif use_bleu:
        output_dir = output_dir + f"bleu-{exp_name}"
    else: 
        output_dir = output_dir + f"{exp_name}"
    
    tags = []
    if select_cosine_threshold:
        tags.append("threshold")
        tags.append(f"cosine{select_cosine_threshold}")
    if use_slem:
        tags.append("slem")
    elif use_bleu:
        tags.append("bleu")
    

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        warmup_ratio=0.2,
        weight_decay=0.01,
        push_to_hub=push_to_hub,
        seed=SEED,
        bf16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        hub_strategy="end"
    )

    data_args = {
        "max_seq_len": 416,
        "doc_stride": 128,
        "do_lower_case": True,
        "wandb_project": "claq-qa-mrc",
        "model": "airesearch/wangchanberta-base-att-spm-uncased",
        "eval_n_best_size": 20,
        "eval_max_answer_length": 64,
        "squad_v2": False,
        "run_name": exp_name,
        "tags": None if len(tags) == 0 else tags
    }

    return training_args, data_args

def preprocess_function(examples, tokenizer, data_args):
    if not data_args["do_lower_case"]:
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=data_args["max_seq_len"],
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            stride=data_args["doc_stride"]
        )

    else:
        questions = [q.strip().lower() for q in examples["question"]]
        examples["context"] = [x.lower() for x in examples["context"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=data_args["max_seq_len"],
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            stride=data_args["doc_stride"]
        )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def prepare_validation_features(examples, tokenizer, data_args):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    if not data_args["do_lower_case"]:
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=data_args["max_seq_len"],
            stride=data_args["doc_stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    else:
        examples["question"] = [q.lstrip().lower() for q in examples["question"]]
        examples["context"] = [x.lower() for x in examples["context"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=data_args["max_seq_len"],
            stride=data_args["doc_stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, data_args):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -data_args["eval_n_best_size"] - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -data_args["eval_n_best_size"] - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > data_args["eval_max_answer_length"]:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not data_args["squad_v2"]:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

def evaluate_dataset(dataset, trainer, tokenizer, data_args, metric):
    features = dataset.map(lambda x: prepare_validation_features(x, tokenizer, data_args), batched=True, remove_columns=dataset.column_names)
    raw_predictions = trainer.predict(features)

    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    predictions = postprocess_qa_predictions(dataset, features, raw_predictions.predictions, tokenizer, data_args)

    if data_args["squad_v2"]:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    # References does not take in other keys at all (thus we need to explicitly select keys)
    references = [{"id": ex["id"], "answers": {"answer_start": ex["answers"]["answer_start"], "text": ex["answers"]["text"]}} for ex in dataset]
    return metric.compute(predictions=formatted_predictions, references=references)

def train_eval_model(train_set, val_set, test_set, training_args, data_args):
    if data_args["tags"]:
        wandb.init(project=data_args["wandb_project"], name=data_args["run_name"], tags=data_args["tags"])
    else:
        wandb.init(project=data_args["wandb_project"], name=data_args["run_name"])

    model = AutoModelForQuestionAnswering.from_pretrained(data_args["model"])
    tokenizer = AutoTokenizer.from_pretrained(data_args["model"])

    tokenized_train = train_set.map(lambda x: preprocess_function(x, tokenizer, data_args), batched=True, remove_columns=train_set.column_names)
    tokenized_val = val_set.map(lambda x: preprocess_function(x, tokenizer, data_args), batched=True, remove_columns=val_set.column_names)

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    metric = load_metric("squad_v2" if data_args["squad_v2"] else "squad")

    # Test the model on val set
    val_metric = evaluate_dataset(val_set, trainer, tokenizer, data_args, metric)
    test_metric = evaluate_dataset(test_set, trainer, tokenizer, data_args, metric)

    test_xquad_metric = evaluate_dataset(test_set.filter(lambda x: x["source"] == "xquad"), trainer, tokenizer, data_args, metric)
    test_tydiqa_metric = evaluate_dataset(test_set.filter(lambda x: x["source"] == "tydiqa"), trainer, tokenizer, data_args, metric)

    # Log Scores
    wandb.log({"val_f1": val_metric["f1"], "val_exact_match": val_metric["exact_match"]})
    wandb.log({"test_f1": test_metric["f1"], "test_exact_match": test_metric["exact_match"]})
    wandb.log({"test_xquad_f1": test_xquad_metric["f1"], "test_xquad_exact_match": test_xquad_metric["exact_match"]})
    wandb.log({"test_tydiqa_f1": test_tydiqa_metric["f1"], "test_tydiqa_exact_match": test_tydiqa_metric["exact_match"]})

    # Cleanup
    wandb.finish()
    del trainer
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-augment-idx", type=str, default=None, help="Continue from a specific augment index")
    parser.add_argument("--from-augment-ratio", type=float, default=None, help="Continue from a specific augment ratio")
    parser.add_argument("--warnings", action="store_true", help="Enable warnings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Dry run only")
    parser.add_argument("--use-slem", action="store_true", help="Use SLEM metric instead")
    parser.add_argument("--use-bleu", action="store_true", help="Use topk bleu instead of topk cosine for running dataset benchmark")
    parser.add_argument("--select-cosine-threshold", type=float, help="Select specific cosine threshold for running dataset benchmark")
    args = parser.parse_args()

    SEED = int(args.seed)
    print(f"Using seed {SEED}")
    seed_everything(SEED)

    # Make models/ folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    if not args.dry_run:
        if not args.warnings:
            warnings.filterwarnings("ignore")

        if args.from_augment_idx:
            print(f"Continuing from {args.from_augment_idx}")
            all_augment_cols = all_augment_cols[int(args.from_augment_idx):]
            print(f"New augment columns: {all_augment_cols}")
        
        for col in tqdm(all_augment_cols):
            gc.collect()
            if col:
                for ratio in range(1, 11):
                    ratio = ratio / 10

                    # Skip if we are continuing from a specific augment index
                    if args.from_augment_idx and all_augment_cols[0] == col and args.from_augment_ratio:
                        if float(args.from_augment_ratio) > ratio:
                            print(f"Skipping {col} {ratio}")
                            continue

                    exp_name = f"{col}_"
                    if args.select_cosine_threshold:
                        exp_name += f"{args.select_cosine_threshold}-"
                        if args.use_bleu:
                            exp_name += f"bleu_{ratio}"
                        elif args.use_slem:
                            exp_name += f"slem_{ratio}"
                        else:
                            raise ValueError("Must specify either --use-bleu or --use-slem if --select-cosine-threshold is specified")
                    else: 
                        exp_name += f"{ratio}"

                    train_set, val_set, test_set = get_ds(col, aug_ratio=ratio, return_hf=True, use_slem=args.use_slem, use_bleu=args.use_bleu, select_cosine_threshold=args.select_cosine_threshold)
                    training_args, data_args = get_training_args(exp_name, use_slem=args.use_slem, use_bleu=args.use_bleu, select_cosine_threshold=args.select_cosine_threshold)

                    train_eval_model(train_set, val_set, test_set, training_args, data_args)

                    # Delete the model to save memory
                    time.sleep(60) # 60 seconds to allow model to be pushed to remote first
                    shutil.rmtree(training_args.output_dir)

            else:
                train_set, val_set, test_set = get_ds(return_hf=True)
                training_args, data_args = get_training_args("original")
                train_eval_model(train_set, val_set, test_set, training_args, data_args)

                # Delete the model to save memory
                time.sleep(60) # 60 seconds to allow model to be pushed to remote first
                shutil.rmtree(training_args.output_dir)