import json
import os
import re
from copy import deepcopy
from enum import Enum
from logging import INFO
from typing import Dict, List, Set, Tuple

import lightning as L
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import get_dataset_split_names, load_dataset, load_dataset_builder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..utils import DatasetInUse, TokenType, setup_logger


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetInUse,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        cache_location="./.cache/",
    ):
        super().__init__()
        self.logger = setup_logger("DataModule", INFO)
        self.dataset = dataset
        self.rel_dict = {}  # Keeps indexes
        self.cache_loc = cache_location
        self.tokenizer = tokenizer
        if not os.path.exists(self.cache_loc):
            os.makedirs(self.cache_loc)
        self.cache_paths = {
            "train": os.path.join(cache_location, "train.parquet"),
            "val": os.path.join(cache_location, "val.parquet"),
            "test": os.path.join(cache_location, "test.parquet"),
        }
        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.metadata = {}

        # I do this more out of need honestly
        self.pre_prepare_data()

    def pre_prepare_data(self):
        """
        prepare_data is called by the Lightning framework.
        We need to have metadata loaded before that though.
        Thus this function. Obviously call this with only one process.
        """
        check_1 = [
            os.path.exists(loc) and os.path.isfile(loc)
            for loc in self.cache_paths.values()
        ] + [os.path.exists(os.path.join(self.cache_loc, "metadata.json"))]
        if all(check_1):
            self.logger.info("📂 Loading cached dataset")
            train_dataset_df = pd.read_parquet(self.cache_paths["train"])
            val_dataset_df = pd.read_parquet(self.cache_paths["test"])
            with open(os.path.join(self.cache_loc, "metadata.json"), "r") as f:
                self.metadata = json.load(f)
            self.logger.info(
                f"We are considering {len(self.metadata['relationships'])} relationships"
            )
        else:
            self.logger.info("🛠 No cached dataset foud. Will build from scratch...")
            train_dataset_df, val_dataset_df, test_dataset_df = self._load_raw_dataset(
                self.dataset, self.tokenizer
            )
        self.logger.info("🚦Preprocessing data, this takes a minute.")
        self.train_dataset, self.val_dataset = self.preprocess_loaded_data(
            train_dataset_df, val_dataset_df, ["tokens", "triplets", "token_types"]
        )
        # self.train_dataset = self.train_dataset.sample(frac=1).reset_index(drop=True)
        # self.val_Dataset = self.val_dataset.sample(frac=1).reset_index(drop=True)
        self.logger.info("Done with data preprocessing.")
        self.train_dataset = train_dataset_df[
            ["tokens", "triplets", "token_types", "ref_text", "ref_raw_triplets"]
        ].values.tolist()
        self.val_dataset = val_dataset_df[
            ["tokens", "triplets", "token_types", "ref_text", "ref_raw_triplets"]
        ].values.tolist()

    def preprocess_loaded_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, selected_columns: List[str]
    ):
        processed_train = train_df.apply(self._preprocess_row, axis=1)
        train_df[["tokens", "triplets", "token_types"]] = processed_train[
            selected_columns
        ].values
        processed_test = test_df.apply(self._preprocess_row, axis=1)
        test_df[["tokens", "triplets", "token_types"]] = processed_test[
            selected_columns
        ].values

        return train_df, test_df

    def _preprocess_row(self, row):
        # Iterate over rows:
        tokens = torch.LongTensor(np.array(row["tokens"]))
        amnt_rels = len(self.metadata["relationships"])
        vocab_size = self.tokenizer.vocab_size

        token_types = torch.LongTensor(np.array(row["token_types"]))
        triplets = torch.LongTensor(np.array(row["triplets"]))

        mask_rel = token_types == TokenType.RELATIONSHIP
        mask_copy = token_types == TokenType.COPY

        # 👀 Careful that this induces an order already
        triplets[mask_rel] += vocab_size
        triplets[mask_copy] += vocab_size + amnt_rels

        return pd.Series(
            [tokens, triplets, token_types], index=["tokens", "triplets", "token_types"]
        )

    def prepare_data(self):
        # Load Stuff
        needs_to_load_data = [
            self.train_dataset == None,
            self.val_dataset == None,
            len(self.metadata) == 0,
        ]
        if any(needs_to_load_data):
            self.pre_prepare_data()

    # Overwrites
    def train_dataloader(self):
        if self.train_dataloader == None:
            raise ValueError("DataModule not prepared. Please first run prepare_data()")
        return DataLoader(
            self.train_dataset,  # type:ignore
            batch_size=self.batch_size,
            num_workers=12,
            collate_fn=collate_fn,
            shuffle=True,
        )

    # Overwrites
    def val_dataloader(self):
        if self.val_dataset == None:
            raise ValueError("DataModule not prepared. Please first run prepare_data()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=12,
            collate_fn=collate_fn,
        )

    def _load_raw_dataset(
        self,
        dataset_type: DatasetInUse,
        tokenizer: PreTrainedTokenizer,
    ):
        self.metadata = {}
        dataset = None
        dfs = {}
        rels_dict = {}
        schema = pa.schema(
            [
                pa.field("tokens", pa.list_(pa.int64())),
                pa.field("triplets", pa.list_(pa.int64())),
                pa.field("token_types", pa.list_(pa.int64())),
                pa.field("ref_text", pa.string()),
                pa.field("ref_raw_triplets", pa.list_(pa.string())),
            ]
        )
        # Method provided by libraries
        dataset = load_dataset("web_nlg", "release_v3.0_en")

        # We will mix them because relationships are not equally spread
        # 👁️ Pay attention here: The split they provide will not ensure all relationships are equally spread across folds.
        #    Therefore, the `add_RathThan_Rem` variable will be used to stop adding relationships that do not exist to non-training folds.

        train = dataset["train"]  # type:ignore
        val = dataset["test"]  # type:ignore
        test = dataset["dev"]  # type:ignore

        bois = {"train": train, "val": val, "test": test}
        current_sample_list = []

        for k, new_samples in bois.items():
            add_RathThan_Rem = True if k == "train" else False
            new_samples, rels = parse_webnlg_ds(
                new_samples, k, tokenizer, rels_dict, add_RathThan_Rem
            )
            current_sample_list += new_samples
            rels_dict.update(rels)
            # Cache this as parquet
            df = pd.DataFrame(
                new_samples,
                columns=[
                    "tokens",
                    "triplets",
                    "token_types",
                    "ref_text",
                    "ref_raw_triplets",
                ],
            )
            dfs[k] = df
            table = pa.Table.from_pandas(df, schema=schema)
            pq.write_table(table, self.cache_paths[k])

        # Make sure that

        # Store a JSon of all metadata:
        self.metadata["relationships"] = list(rels_dict.keys())
        metadata_js = json.dumps(self.metadata)
        with open(os.path.join(self.cache_loc, "metadata.json"), "w") as f:
            f.write(metadata_js)

        # We make df out of train, val, test
        train = pd.read_parquet(self.cache_paths["train"])
        return dfs["train"], dfs["val"], dfs["test"]  # type:ignore


def parse_webnlg_ds(  # TODO: clean this method up
    ds,
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    seen_rels: Dict[str, int],
    add_RathThan_Rem=True,
    max_length=1024,
) -> Tuple[List[Tuple], Dict[str, int]]:  # HACK: remove ths hardcode max_length
    result = []
    print("Done")

    rel_dict = deepcopy(seen_rels)
    # TODO: maybe make this variable be passed to the class
    output_max_len = 256

    bar = tqdm(total=len(ds), desc=f'Going through dataset "{dataset_name}"')
    for row in ds:
        # Change Triplets into an eaier to read format
        dirty_triplets = row["modified_triple_sets"]["mtriple_set"][0]

        # Get Matching Text
        for i in range(len(row["lex"]["comment"])):
            # Sanitization
            if row["lex"]["comment"][i] == "bad":
                continue
            if len(row["lex"]["text"][i]) >= max_length:
                continue

            text_examples = row["lex"]["text"]
            # Start appending examples
            for text in text_examples:
                text = clean_string(text)
                # Tokenize the incoming text
                tokd_text = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                )["input_ids"]

                tokd_triplets, updated_rel_dict, token_types = get_target_encoding(
                    text, dirty_triplets, tokenizer, rel_dict, add_RathThan_Rem
                )
                if len(tokd_triplets) == 0 and len(token_types) == 0:
                    continue
                rel_dict.update(updated_rel_dict)
                # Tokenize them and remove the outer stuff

                # Add Padding to said triplets
                amnt_pad = output_max_len - len(tokd_triplets)
                tokd_triplets += [tokenizer.pad_token_id] * amnt_pad
                token_types += [TokenType.NORMAL.value] * amnt_pad

                assert (
                    amnt_pad > 0
                ), f"Padding assumptions are wrong.:{tokenizer.decode(tokd_triplets)}"  # type: ignore

                assert (
                    len(tokd_text) == tokenizer.model_max_length  # type: ignore
                ), f" Tokenized Text of not proper length ({len(tokd_text)}):{tokenizer.decode(tokd_text)}"  # type: ignore

                assert (
                    len(tokd_triplets) == output_max_len
                ), f"Wrong length for tokd_triplets"

                result.append(
                    [
                        tokd_text,
                        tokd_triplets,
                        token_types,
                        # For Reference
                        text,
                        dirty_triplets,
                    ]
                )
        bar.update(1)

    return result, rel_dict


def get_target_encoding(
    sentence: str,
    dtriplets: List[str],
    tokenizer: PreTrainedTokenizer,
    rel_dict: Dict,
    add_RathThan_Rem: bool,
) -> Tuple[List[int], Dict[str, int], List[int]]:
    """
    An alternate (and possibly final) approach to extracting triplets.
    Sub-Obj are not tokenized, but rather given an index corresponding to input sentence.
    Note
    ----
        token_ids_types is to distinguish between: copy, relationship, and normal (vocab) tokens
    """
    # Split the entity into words
    token_ids_types = []
    new_triplets = []
    unique_rels = deepcopy(rel_dict)

    for i, triplet in enumerate(dtriplets):
        # NOTE: maybe try to use `tokenizer.tokenize` for more fine grained splitting ?
        # Replace _ with " "
        triplet = clean_string(triplet)
        sentence = clean_string(sentence)

        trip = [t.strip() for t in triplet.split("|")]
        rel = trip[1]
        e1 = tokenizer.encode(
            trip[0], add_special_tokens=False, is_split_into_words=True
        )
        e2 = tokenizer.encode(
            trip[2], add_special_tokens=False, is_split_into_words=True
        )
        sentence_words = tokenizer.encode(sentence)

        best_e1 = find_consecutive_largest(sentence_words, e1)
        best_e2 = find_consecutive_largest(sentence_words, e2)

        if best_e1 == None or best_e2 == None:
            return [], unique_rels, []

        # Add to Dictionary
        if rel not in unique_rels.keys():
            if add_RathThan_Rem:
                unique_rels[rel] = len(unique_rels.keys())
            else:  # This is non-trainign dataset so we dont add this
                continue

        cur_len = lambda: len(new_triplets)

        new_triplets += tokenizer.convert_tokens_to_ids(["<s>"])  # type: ignore
        token_ids_types += [TokenType.NORMAL.value] * cur_len()
        length_sofar = cur_len()

        # Add Relationship
        new_triplets += [unique_rels[rel]]
        token_ids_types += [TokenType.RELATIONSHIP.value]
        length_sofar = cur_len()

        # Comma Separator
        new_triplets += tokenizer.encode(", ", add_special_tokens=False, is_split_into_words=True)  # type: ignore
        token_ids_types += [TokenType.NORMAL.value] * (cur_len() - length_sofar)
        length_sofar = cur_len()

        # Add Copy Subject 1
        new_triplets += np.arange(best_e1[0], best_e1[-1] + 1).tolist()
        token_ids_types += [TokenType.COPY.value] * (cur_len() - length_sofar)
        length_sofar = cur_len()

        # Comma Separator
        new_triplets += tokenizer.encode(", ", add_special_tokens=False, is_split_into_words=True)  # type: ignore
        token_ids_types += [TokenType.NORMAL.value] * (cur_len() - length_sofar)
        length_sofar = cur_len()

        # Add Copy Subject 2
        new_triplets += np.arange(best_e2[0], best_e2[-1] + 1).tolist()
        token_ids_types += [TokenType.COPY.value] * (cur_len() - length_sofar)
        length_sofar = cur_len()

        # Final Separator
        if i != len(dtriplets) - 1:
            new_triplets += tokenizer.encode(
                " | ", add_special_tokens=False, is_split_into_words=True
            )
        else:
            new_triplets += tokenizer.convert_tokens_to_ids(["</s>"])
        token_ids_types += [TokenType.NORMAL.value] * (cur_len() - length_sofar)

        # "Flatten the whole list):
    return new_triplets, unique_rels, token_ids_types


def clean_string(str):
    return (
        str.replace('"', "")
        .replace("'", "")
        .replace(",", "")
        .replace(".", "")
        .replace("_", " ")
    )


def find_consecutive_largest(sentence_words, entity_words):
    i, j = 0, 0
    record = []
    best_shot = []
    while i < len(sentence_words):
        if j >= len(entity_words):
            break
        if sentence_words[i] != entity_words[j]:
            if len(record) > len(best_shot):
                best_shot = record
            j = 0
            record = []
        else:
            record.append(i)
            j += 1
        i += 1
    if len(record) > len(best_shot):
        best_shot = record
    if len(best_shot) == 0:
        best_shot = None
    return best_shot


def collate_fn(batch):
    tknd_sentence, target, token_types, ref_text, ref_raw_triplets = zip(*batch)
    if isinstance(tknd_sentence[0], str):
        print("wut")
    tknd_sentence = torch.stack(tknd_sentence)
    target = torch.stack(target)
    # references = zip(ref_text, ref_)
    # token_types = torch.stack(token_types)
    return tknd_sentence, target, ref_text, ref_raw_triplets  # , token_types


### Old Code probably useless ###

# def _fix_entity_for_copymechanism_0(
#     self, sentence: str, dtriplets: List[str]
# ) -> Tuple[List[List], Set[str]]:
#     # Split the entity into words
#     new_triplets = []
#     unique_rels = set()
#     for i, triplet in enumerate(dtriplets):
#         # NOTE: maybe try to use `tokenizer.tokenize` for more fine grained splitting ?
#         trip = [t.strip() for t in triplet.split("|")]
#         rel = trip[1]
#         e1 = re.split("_|\s", trip[0])
#         e2 = re.split("_|\s", trip[2])
#         sentence_words = re.split(" |,|\.", sentence)
#         sentence_words = [sw for sw in sentence_words if sw != ""]
#
#         best_e1 = find_consecutive_largest(sentence_words, e1)
#         best_e2 = find_consecutive_largest(sentence_words, e2)
#         if best_e1 == None or best_e2 == None:
#             return [[]], unique_rels
#
#         # Add to Dictionary
#         if rel not in self.rel_dict.keys():
#             self.rel_dict[rel] = len(self.rel_dict.keys())
#             self.local_rels.update(rel)
#
#         unique_rels.add(rel)
#         new_triplet = [
#             " ".join(sentence_words[best_e1[0] : best_e1[-1] + 1]),
#             trip[1],
#             " ".join(sentence_words[best_e2[0] : best_e2[-1] + 1]),
#         ]
#
#         new_triplets.append(new_triplet)
#     return new_triplets, unique_rels
#
# def _tokenize_triplets_joint(self, triplets: List, tokenizer: PreTrainedTokenizer):
#     result = [tokenizer.convert_tokens_to_ids("<s>")]
#     for i, triplet in enumerate(triplets):
#         if len(triplet) == 0:
#             continue
#         result += tokenizer.encode(triplet[0] + " ", add_special_tokens=False)
#         result += tokenizer.encode(triplet[1] + " ", add_special_tokens=False)
#         result += tokenizer.encode(triplet[2], add_special_tokens=False)
#         if i != len(triplets) - 1:
#             result += tokenizer.encode(",", add_special_tokens=False)
#     result += tokenizer.convert_tokens_to_ids("</s>")
#     return result
#
# def _tokenize_triplets(self, triplets: List, tokenizer: PreTrainedTokenizer):
#     new_ones = []
#     for triplet in triplets:
#         if len(triplet) == 0:
#             continue
#         new_ones.append(
#             [
#                 tokenizer.encode(triplet[0], add_special_tokens=False),
#                 [self.rel_dict[triplet[1]]],
#                 tokenizer.encode(triplet[2], add_special_tokens=False)[1:-1],
#             ]
#         )
#     return new_ones
#
# def _fix_entity_for_copymechanism_1(
#     self, sentence: str, dtriplets: List[str]
# ) -> Tuple[List[List], Set[str]]:
#     """
#     An alternate approach to extracting triplets.
#     Sub-Obj are not tokenized, but rather given an index corresponding to input sentence.
#     (unused)
#     """
#     # Split the entity into words
#     new_triplets = []
#     unique_rels = set()
#     for i, triplet in enumerate(dtriplets):
#         # NOTE: maybe try to use `tokenizer.tokenize` for more fine grained splitting ?
#         trip = [t.strip() for t in triplet.split("|")]
#         rel = trip[1]
#         e1 = re.split("_|\s", trip[0])
#         e2 = re.split("_|\s", trip[2])
#         sentence_words = re.split(" |,|\.", sentence)
#         sentence_words = [sw for sw in sentence_words if sw != ""]
#
#         best_e1 = find_consecutive_largest(sentence_words, e1)
#         best_e2 = find_consecutive_largest(sentence_words, e2)
#         if best_e1 == None or best_e2 == None:
#             return [[]], unique_rels
#
#         # Add to Dictionary
#         if rel not in self.rel_dict.keys():
#             self.rel_dict[rel] = len(self.rel_dict.keys())
#             self.local_rels.update(rel)
#
#         unique_rels.add(rel)
#         # Encode positions rather than actual tokens
#         new_triplet = [self.rel_dict[rel]]
#         new_triplet += (-1 * (1 + np.arange(best_e1[0], best_e1[-1] + 1))).tolist()
#         new_triplet += (-1 * (1 + np.arange(best_e2[0], best_e2[-1] + 1))).tolist()
#         # "Flatten the whole list):
#
#         new_triplets.append(new_triplet)
#     return new_triplets, unique_rels
