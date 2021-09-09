
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import itertools
import collections
import tensorflow as tf
import datetime

import numpy as np
import datasets as hf_datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2Config,
    GPT2ForSequenceClassification,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# from transformers.trainer_utils import is_main_process

# import pyarrow.csv
from datasets import Features, ClassLabel, Value
task_to_keys = {
    "cls": ("sentence", None),
    "pawsx": ("sentence1", "sentence2"),
    "xnli": ("premise", "hypothesis"),
    "wsd": ("sentence", None),
}

task_to_metrics = {
    "cls": "sst2",
    "pawsx": "mrpc",
    "xnli": "mnli",
    "wsd": "",
}

DevResult = collections.namedtuple('DevResult', 'seed, learning_rate batch_size eval_metric_1 eval_metric_2 eval_loss')

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    n_seeds: Optional[int] = field(
        default=5,
        metadata={"help": "Number of run for task with small training sets."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    predict_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    learning_rates: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    batch_sizes: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if True else logging.WARN,  # is_main_process(training_args.local_rank)
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if True:  # is_main_process(training_args.local_rank)
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        # transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    hyper_parameters_gridsearch = {
        'learning_rate': [float(lr) for lr in data_args.learning_rates.split('/')],
        'batch_size': [float(lr) for lr in data_args.batch_sizes.split('/')],
    }

    hyper_parameters_values = [p for p in itertools.product(*[*hyper_parameters_gridsearch.values()])]

    # Set seed before initializing model.
    # set_seed(training_args.seed)

    if data_args.task_name in ['pawsx', 'xnli', 'wsd']:
        n_seeds_ = 1
    else:
        n_seeds_ = data_args.n_seeds

    eval_results = {}
    test_results = {}

    for hyperparameters in hyper_parameters_values:

        for seed in range(n_seeds_):

            training_args.learning_rate = hyperparameters[0]
            training_args.per_device_train_batch_size = int(hyperparameters[1])
            print('\nstart hyper-parameters search with : lr: {} and batch_size: {} without seed {}'.format(
                training_args.learning_rate, training_args.per_device_train_batch_size, seed))

            def split_sentences(example):
                example['sentence2'] = example['sentence1'].split('\t')[1]
                example['sentence1'] = example['sentence1'].split('\t')[0]
                return example

            def cast_labels(example):
                example['premise'] = example['premise'].strip('\"')
                example['label'] = example['hypothesis']
                example['hypothesis'] = example['premise'].split('\t')[1]
                example['premise'] = example['premise'].split('\t')[0]
                if example['label'] == "entailment":
                    example['label'] = 0
                elif example['label'] == "neutral":
                    example['label'] = 1
                elif example['label'] == "contradiction":
                    example['label'] = 2
                return example

            if data_args.task_name == "pawsx":
                datasets = load_dataset("csv", data_files={"train": data_args.train_file,
                                                           "validation": data_args.validation_file,
                                                           "test": data_args.predict_file},
                                        column_names=['label', 'idx1', 'idx2', 'sentence1', 'sentence2'],
                                        skiprows=0, sep='\t')
                # datasets = hf_datasets.load_from_disk(data_args.train_file)
                datasets = datasets.map(split_sentences)
                datasets.cast_(Features({'label': ClassLabel(num_classes=2),
                                         'idx1': Value(dtype='int64'),
                                         'idx2': Value(dtype='int64'),
                                         'sentence1': Value(dtype='string'),
                                         'sentence2': Value(dtype='string'),
                                         }))
            elif data_args.task_name == "xnli":
                datasets = load_dataset("csv", data_files={"train": data_args.train_file,
                                                           "validation": data_args.validation_file,
                                                           "test": data_args.predict_file},
                                        column_names=['premise', 'hypothesis', 'label'], skiprows=1, sep='\t')
                # datasets = hf_datasets.load_from_disk(data_args.train_file)
                datasets = datasets.map(cast_labels)
                datasets.cast_(Features({'premise': Value(dtype='string'),
                                         'hypothesis': Value(dtype='string'),
                                         'label': ClassLabel(names=['entailment', 'neutral', 'contradiction']),
                                         }))
            elif data_args.task_name == "cls":
                datasets = load_dataset("csv", data_files={"train": data_args.train_file,
                                                           "validation": data_args.validation_file,
                                                           "test": data_args.predict_file},
                                        column_names=['sentence', 'label'], skiprows=1, sep='\t')
                # datasets = hf_datasets.load_from_disk(data_args.train_file)
                datasets.cast_(Features({'sentence': Value(dtype='string'),
                                         'label': ClassLabel(num_classes=2),
                                         }))

            # Labels
            if data_args.task_name is not None:
                is_regression = data_args.task_name == "stsb"
                if not is_regression:
                    label_list = datasets["train"].features["label"].names
                    num_labels = len(label_list)
                else:
                    num_labels = 1
            else:
                # Trying to have good defaults here, don't hesitate to tweak to your needs.
                is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
                if is_regression:
                    num_labels = 1
                else:
                    # A useful fast method:
                    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                    label_list = datasets["train"].unique("label")
                    label_list.sort()  # Let's sort it for determinism
                    num_labels = len(label_list)

            # Load pretrained model and tokenizer
            #
            # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
            # download model & vocab.
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=task_to_metrics[data_args.task_name],
                output_attention=False,
                output_hidden_states=False,
                use_cache=False
                # cache_dir=model_args.cache_dir,
            )

            tokenizer = GPT2Tokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
            )
            # if tokenizer.__class__.__name__ == 'GPT2TokenizerFast':
            tokenizer.add_special_tokens({
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>"
            })
            # config = GPT2Config(
            #     vocab_size=tokenizer.vocab_size,
            #     bos_token_id=tokenizer.bos_token_id,
            #     eos_token_id=tokenizer.bos_token_id,
            #     n_embd=120,  # 1200, 1536
            #     n_layer=1,  # 36, 40
            #     n_head=3,  # 12, 16
            #     output_attentions=False,
            #     output_hidden_states=False,
            #     use_cache=False,
            #     num_labels=num_labels,
            #     finetuning_task=task_to_metrics[data_args.task_name],
            # )
            config.pad_token_id = tokenizer.pad_token_id

            # model = GPT2ForSequenceClassification(config)
            model = GPT2ForSequenceClassification.from_pretrained(  # AutoModelForSequenceClassification
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                # cache_dir=model_args.cache_dir,
            )
            model.train()

            # Preprocessing the datasets
            if data_args.task_name is not None:
                sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
            else:
                # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
                non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
                if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                    sentence1_key, sentence2_key = "sentence1", "sentence2"
                else:
                    if len(non_label_column_names) >= 2:
                        sentence1_key, sentence2_key = non_label_column_names[:2]
                    else:
                        sentence1_key, sentence2_key = non_label_column_names[0], None

            # Padding strategy
            if data_args.pad_to_max_length:
                padding = "max_length"
                max_length = data_args.max_seq_length
            else:
                # We will pad later, dynamically at batch creation, to the max sequence length in each batch
                padding = False
                max_length = None

            # Some models have set the order of the labels to use, so let's make sure we do use it.
            label_to_id = None
            if (
                    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
                    and data_args.task_name is not None
                    and is_regression
            ):
                # Some have all caps in their config, some don't.
                label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
                if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                    label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
                else:
                    logger.warn(
                        "Your model seems to have been trained with labels, but they don't match the dataset: ",
                        f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                        "\nIgnoring the model labels as a result.",
                    )
            elif data_args.task_name is None:
                label_to_id = {v: i for i, v in enumerate(label_list)}

            def preprocess_function(examples):
                # Tokenize the texts
                args = (
                    (examples[sentence1_key],) if sentence2_key is None else (
                        examples[sentence1_key], examples[sentence2_key])
                ) # examples[sentence2_key]
                result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

                # Map labels to IDs (not necessary for GLUE tasks)
                if label_to_id is not None and "label" in examples:
                    result["label"] = [label_to_id[l] for l in examples["label"]]
                return result

            datasets = datasets.map(preprocess_function, batched=True,
                                    load_from_cache_file=not data_args.overwrite_cache)

            train_dataset = datasets["train"]
            eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
            if data_args.task_name is not None:
                test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

            # Get the metric function
            if data_args.task_name is not None:
                metric = load_metric("/content/glue_metrics.py", task_to_metrics[data_args.task_name])

            # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
            # compute_metrics

            # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
            # predictions and label_ids field) and has to return a dictionary string to float.
            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
                if data_args.task_name is not None:
                    result = metric.compute(predictions=preds, references=p.label_ids)
                    if len(result) > 1:
                        result["combined_score"] = np.mean(list(result.values())).item()
                    return result
                elif is_regression:
                    return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
                else:
                    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

            if not tf.io.gfile.exists(
                    os.path.join(training_args.output_dir, model_args.model_name_or_path.split('/')[-1])):
                tf.io.gfile.makedirs(
                    os.path.join(training_args.output_dir, model_args.model_name_or_path.split('/')[-1]))
                tf.io.gfile.makedirs(
                    os.path.join(training_args.output_dir, model_args.model_name_or_path.split('/')[-1], 'flue'))
                tf.io.gfile.makedirs(
                    os.path.join(training_args.output_dir, model_args.model_name_or_path.split('/')[-1], 'flue', data_args.task_name))

            output_dir = '{}/{}/flue/{}/{}_{}_{}'.format(
                training_args.output_dir,
                model_args.model_name_or_path.split('/')[-1],
                data_args.task_name,
                str(seed),
                str(training_args.learning_rate),
                str(training_args.train_batch_size))

            training_args.output_dir = output_dir
            training_args.save_steps = 10000
            training_args.save_total_limit = 2

            # Initialize our Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
                data_collator=default_data_collator if data_args.pad_to_max_length else None,
            )

            # Training
            if training_args.do_train:
                trainer.train()
                # trainer.train(
                #     model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
                # )
                trainer.save_model()  # Saves the tokenizer too for easy upload

            # Evaluation

            if training_args.do_eval:
                logger.info("*** Evaluate ***")

                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                test_result = trainer.evaluate(eval_dataset=test_dataset)

                dev_metric_1 = eval_result['eval_accuracy']
                test_metric_1 = test_result['eval_accuracy']
                if data_args.task_name in ['WSD']:
                    dev_metric_2 = eval_result['f1']
                    test_metric_2 = test_result['f1']
                else:
                    dev_metric_2 = 0
                    test_metric_2 = 0
                eval_results[output_dir] = DevResult(seed, training_args.learning_rate, training_args.train_batch_size,
                                                    dev_metric_1, dev_metric_2,
                                                    eval_result['eval_loss'])
                test_results[output_dir] = DevResult(seed, training_args.learning_rate, training_args.train_batch_size,
                                                     test_metric_1, test_metric_2,
                                                     test_result['eval_loss'])

                if trainer.is_world_process_zero():
                    logger.info(f"***** Eval results {data_args.task_name} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")

    if training_args.do_predict:
        logger.info("*** Test ***")

        best_result = -1
        for k, v in eval_results.items():
            if v.eval_metric_1 > best_result:
                best_result = v.eval_metric_1
                best_result_2 = v.eval_metric_2
                best_estimator_output_dir = k
                best_estimator_learning_rate = v.learning_rate
                best_estimator_train_batch_size = v.batch_size
                best_seed = v.seed

        same_seed_results_1 = []
        same_seed_results_2 = []
        for k, v in eval_results.items():
            if (best_estimator_learning_rate == v.learning_rate) and (best_estimator_train_batch_size == v.batch_size):
                same_seed_results_1.append(v.eval_metric_1)
                same_seed_results_2.append(v.eval_metric_2)

        print('\nhyper-parameters: seed: {} lr: {} and batch size: {} saved in dir: {}'
              .format(best_seed, best_estimator_learning_rate, best_estimator_train_batch_size,
                      best_estimator_output_dir))

        print('\n best dev results 1: {}, avg: {}, std: {}, best dev results 2: {}, : avg: {}, std: {}'.format(
            round(best_result * 100, 1), round(np.mean(same_seed_results_1) * 100, 1),
            round(np.std(same_seed_results_1) * 100, 1),
            round(best_result_2 * 100, 1),
            round(np.mean(same_seed_results_2) * 100, 1), round(np.std(same_seed_results_2) * 100, 1)))

        best_test_results = test_results[best_estimator_output_dir]
        logger.info(f"***** Test results {data_args.task_name} *****")
        logger.info(f"  eval_metric_1 = {best_test_results.eval_metric_1}")
        logger.info(f"  eval_metric_2 = {best_test_results.eval_metric_2}")
        print("***** Test results {} *****".format(data_args.task_name))
        print("  eval_metric_1 = {}".format(best_test_results.eval_metric_1))
        print("  eval_metric_2 = {}".format(best_test_results.eval_metric_2))


    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
