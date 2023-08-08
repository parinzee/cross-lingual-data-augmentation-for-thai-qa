import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, DefaultDataCollator
from utils import seed_everything
from benchmark_qa import get_ds, prepare_validation_features, postprocess_qa_predictions, get_training_args

def evaluate_model(model, tokenizer, dataset):
    training_args, data_args = get_training_args("mcnemar-test", push_to_hub=False)
    
    # Convert the dataset into a format suitable for the model
    features = dataset.map(lambda x: prepare_validation_features(x, tokenizer, data_args), batched=True, remove_columns=dataset.column_names)
    
    # Use the model to make predictions
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    raw_predictions = trainer.predict(features)
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    
    # Postprocess the predictions
    predictions = postprocess_qa_predictions(dataset, features, raw_predictions.predictions, tokenizer, data_args)

     # Create a dictionary of references keyed by ID
    references = {ex["id"]: ex["answers"]["text"][0] for ex in dataset}

    # Compare the predictions to the ground truth using IDs
    results = [int(predictions[pred_id] == references[pred_id]) for pred_id in tqdm(predictions, total=len(predictions), desc="Evaluating")]
    
    return results

def main(args):
    # Load the models
    model_1 = AutoModelForQuestionAnswering.from_pretrained(args.original_model)
    model_2 = AutoModelForQuestionAnswering.from_pretrained(args.model)

    # Load the corresponding tokenizer
    tokenizer_1 = AutoTokenizer.from_pretrained(args.original_model)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.model)

    # Load the test dataset
    _, _, test_dataset = get_ds(return_hf=True)

    # Evaluate the models
    results_model_1 = evaluate_model(model_1, tokenizer_1, test_dataset)
    results_model_2 = evaluate_model(model_2, tokenizer_2, test_dataset)

    # Compute McNemar's test
    # The input should be a 2x2 contingency table
    table = pd.crosstab(results_model_1, results_model_2)
    result = mcnemar(table, exact=True)

    return result.pvalue, result.statistic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--original-model", type=str, required=False, default="parinzee/claq-qa-th-wangchanberta-original", help="Path to the original model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    p_value, statistic = main(args)
    print(f"p-value: {p_value}")
    print(f"statistic: {statistic}")
