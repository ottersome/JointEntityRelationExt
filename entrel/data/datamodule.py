import json
import os
import re
from logging import INFO
from typing import List, Set, Tuple

import lightning as L
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_split_names, load_dataset, load_dataset_builder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..utils import DatasetInUse, setup_logger


class DataModule(L.LightningDataModule):
    def __init__(
        self, dataset: DatasetInUse, batch_size: int, cache_location="./.cache/"
    ):
        self.logger = setup_logger("DataModule", INFO)
        self.dataset = dataset
        self.rel_dict = {}  # Keeps indexes
        self.cache_loc = cache_location
        if not os.path.exists(self.cache_loc):
            os.makedirs(self.cache_loc)
        self.cache_paths = {
            "train": os.path.join(cache_location, "train.parquet"),
            "val": os.path.join(cache_location, "val.parquet"),
            "test": os.path.join(cache_location, "test.parquet"),
        }
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.metadata = {}

    def prepare_data(self, tokenizer: PreTrainedTokenizer):
        # Load Stuff
        # dsf = DatasetFactory(self.dataset)
        # Check if
        check = [
            os.path.exists(loc) and os.path.isfile(loc)
            for loc in self.cache_paths.values()
        ]
        if all(check):
            self.logger.info("📂 Loading cached dataset")
            train_dataset_df = pd.read_parquet(self.cache_paths["train"])
            test_dataset_df = pd.read_parquet(self.cache_paths["test"])
            with open(os.path.join(self.cache_loc, "metadata.json"), "r") as f:
                self.metadata = json.load(f)
            self.logger.info(
                f"We are considering {len(self.metadata['relationships'])} relationships"
            )

        else:
            self.logger.info("🛠 No cached dataset foud. Will build from scratch...")
            train_dataset_df, test_dataset_df, _ = self._load_raw_dataset(
                self.dataset, tokenizer
            )
        # Datasets so far are pandas dataframes. Before converting to a list
        # want to UNNEST column 'triplets':
        # Now only select columns 'tokens' and 'triplets'. But as simply lists, not numpy
        self.train_dataset = train_dataset_df.values.tolist()
        self.test_dataset = test_dataset_df.values.tolist()

    # Overwrites
    def train_dataloader(self):
        if self.train_dataloader == None:
            raise ValueError("DataModule not prepared. Please first run prepare_data()")
        return DataLoader(
            self.train_dataset,  # type:ignore
            batch_size=self.batch_size,
            num_workers=12,
        )

    # Overwrites
    def val_dataloader(self):
        if self.test_dataset == None:
            raise ValueError("DataModule not prepared. Please first run prepare_data()")
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=12  # type:ignore
        )

    def parse_webnlg_ds(  # TODO: clean this method up
        self, ds, tokenizer: PreTrainedTokenizer, max_length=1024
    ) -> Tuple[List[Tuple], Set[str]]:  # HACK: remove ths hardcode max_length
        result = []
        print("Done")

        skips = 0  # TEST: for statistics
        unique_rels = set()

        bar = tqdm(total=len(ds), desc="Going through dataset")
        for row in ds:
            # Change Triplets into an eaier to read format
            dirty_triplets = row["modified_triple_sets"]["mtriple_set"][0]
            # triplets = []
            self.local_rels = set()

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

                    fixed_triplets, rels = self._fix_entity_for_copymechanism(
                        text, dirty_triplets
                    )
                    unique_rels = unique_rels.union(rels)
                    # Tokenize them and remove the outer stuff
                    if len(fixed_triplets[0]) == 0:
                        skips += 1
                        continue  # We didnt get any matches
                    tokd_triplets = self._tokenize_triplets_joint(
                        fixed_triplets, tokenizer
                    )
                    assert isinstance(
                        tokd_triplets[0], int
                    ), f"The triplet looks like {tokd_triplets}"
                    result.append(
                        [
                            tokd_text,
                            tokd_triplets,
                            text,
                            dirty_triplets,
                            list(self.local_rels),
                        ]
                    )
            bar.update(1)

        self.logger.info(
            f"We ended with {skips} skipped examples and result {len(result)}\n"
            f"Ratio of skips is {skips/(len(result) + skips)}\n"
            f"Amount of unique rels {unique_rels}"
        )
        return result, unique_rels

    def _load_raw_dataset(
        self,
        dataset_type: DatasetInUse,
        tokenizer: PreTrainedTokenizer,
        encoder_max=512,
    ):
        self.metadata = {}
        self.unique_rels = set()
        dataset = None
        dfs = {}
        schema = pa.schema(
            [
                pa.field("tokens", pa.list_(pa.int64())),
                pa.field("triplets", pa.list_(pa.int64())),
                pa.field("ref_text", pa.string()),
                pa.field("ref_raw_triples", pa.list_(pa.string())),
                pa.field("ref_rels", pa.list_(pa.string())),
            ]
        )
        if dataset_type == DatasetInUse.NLG:
            # Method provided by libraries
            dataset = load_dataset("web_nlg", "release_v3.0_en")

            train = dataset["train"]  # type:ignore
            val = dataset["test"]  # type:ignore
            test = dataset["dev"]  # type:ignore

            bois = {"train": train, "val": val, "test": test}

            for k, boi in bois.items():
                boi, rels = self.parse_webnlg_ds(boi, tokenizer)
                self.unique_rels = self.unique_rels.union(rels)
                # Cache this as parquet
                df = pd.DataFrame(
                    boi,
                    columns=[
                        "tokens",
                        "triplets",
                        "ref_text",
                        "ref_raw_triples",
                        "ref_rels",
                    ],
                )
                dfs[k] = df
                # df.to_parquet(self.cache_paths[k])
                table = pa.Table.from_pandas(df, schema=schema)
                pq.write_table(table, self.cache_paths[k])

        # Store a JSon of all metadata:
        self.metadata["relationships"] = list(self.rel_dict.keys())
        metadata_js = json.dumps(self.metadata)
        with open(os.path.join(self.cache_loc, "metadata.json"), "w") as f:
            f.write(metadata_js)

        # We make df out of train, val, test
        train = pd.read_parquet(self.cache_paths["train"])
        return dfs["train"], dfs["val"], dfs["test"]  # type:ignore

    def _fix_entity_for_copymechanism_0(
        self, sentence: str, dtriplets: List[str]
    ) -> Tuple[List[List], Set[str]]:
        # Split the entity into words
        new_triplets = []
        unique_rels = set()
        for i, triplet in enumerate(dtriplets):
            # NOTE: maybe try to use `tokenizer.tokenize` for more fine grained splitting ?
            trip = [t.strip() for t in triplet.split("|")]
            rel = trip[1]
            e1 = re.split("_|\s", trip[0])
            e2 = re.split("_|\s", trip[2])
            sentence_words = re.split(" |,|\.", sentence)
            sentence_words = [sw for sw in sentence_words if sw != ""]

            best_e1 = find_consecutive_largest(sentence_words, e1)
            best_e2 = find_consecutive_largest(sentence_words, e2)
            if best_e1 == None or best_e2 == None:
                return [[]], unique_rels

            # Add to Dictionary
            if rel not in self.rel_dict.keys():
                self.rel_dict[rel] = len(self.rel_dict.keys())
                self.local_rels.update(rel)

            unique_rels.add(rel)
            new_triplet = [
                " ".join(sentence_words[best_e1[0] : best_e1[-1] + 1]),
                trip[1],
                " ".join(sentence_words[best_e2[0] : best_e2[-1] + 1]),
            ]

            new_triplets.append(new_triplet)
        return new_triplets, unique_rels

    def _tokenize_triplets_joint(self, triplets: List, tokenizer: PreTrainedTokenizer):
        result = []
        for i, triplet in enumerate(triplets):
            if len(triplet) == 0:
                continue
            result += tokenizer.encode(triplet[0], add_special_tokens=False)
            result.append(self.rel_dict[triplet[1]])
            result += tokenizer.encode(triplet[2], add_special_tokens=False)
            if i != len(triplets) - 1:
                result += tokenizer.encode(",", add_special_tokens=False)
        return result

    def _tokenize_triplets(self, triplets: List, tokenizer: PreTrainedTokenizer):
        new_ones = []
        for triplet in triplets:
            if len(triplet) == 0:
                continue
            new_ones.append(
                [
                    tokenizer.encode(triplet[0], add_special_tokens=False),
                    [self.rel_dict[triplet[1]]],
                    tokenizer.encode(triplet[2], add_special_tokens=False)[1:-1],
                ]
            )
        return new_ones

    def fix_entity_for_copy_mechanism_1():
        """
        An alternate (and possibly final) approach to extracting triplets.
        Sub-Obj are not tokenized, but rather given an index corresponding to input sentence.
        """
        # Split the entity into words
        new_triplets = []
        unique_rels = set()
        for i, triplet in enumerate(dtriplets):
            # NOTE: maybe try to use `tokenizer.tokenize` for more fine grained splitting ?
            trip = [t.strip() for t in triplet.split("|")]
            rel = trip[1]
            e1 = re.split("_|\s", trip[0])
            e2 = re.split("_|\s", trip[2])
            sentence_words = re.split(" |,|\.", sentence)
            sentence_words = [sw for sw in sentence_words if sw != ""]

            best_e1 = find_consecutive_largest(sentence_words, e1)
            best_e2 = find_consecutive_largest(sentence_words, e2)
            if best_e1 == None or best_e2 == None:
                return [[]], unique_rels

            # Add to Dictionary
            if rel not in self.rel_dict.keys():
                self.rel_dict[rel] = len(self.rel_dict.keys())
                self.local_rels.update(rel)

            unique_rels.add(rel)
            # Encode positions rather than actual tokens
            new_triplet = [
                range(best_e1[0], best_e1[-1] + 1),
                trip[1],
                range(best_e2[0], best_e2[-1] + 1),
            ]

            new_triplets.append(new_triplet)
        return new_triplets, unique_rels


def clean_string(str):
    return str.replace('"', "").replace("'", "").replace(",", "").replace(".", "")


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
