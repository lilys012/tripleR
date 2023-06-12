from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline
import random
import numpy as np
import math

logger = logging.getLogger(__name__)

class GenericDataLoader:
    
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file
    
    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels

    def load(self, dataset_name=None, method=0, generator_name_or_path=None, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data_rev_dict = {0:'arguana', 1:'fiqa', 2:'nfcorpus', 3:'touche', 4:'scifact', 5:'fever'}
            if method == 2: # pseudo-document is revised w/ flan-t5-xl based on doc
                rev_queries = {}
                prompts = [['argument', 'counter argument'], ['passage', 'query'], ['article', 'query'], ['passage', 'debate'], ['passage', 'finding'], ['passage', 'fact'], ['passage', 'question']]
                prompt = f"Revise {prompts[dataset_name][1]} if it contains errors based on {prompts[dataset_name][0]}. "
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
                model.to(device)
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
                batch_size = 32

                print("Revising queries ... ")
                i = 0
                input_texts, input_qids = [], []
                for qid in tqdm(self.qrels):
                    for pid in self.qrels[qid]:
                        if self.qrels[qid][pid] == 1:
                            input_texts.append(prompt + f"{prompts[dataset_name][1]}: " + self.queries[qid] + f" {prompts[dataset_name][0]}: " + self.corpus[pid]["text"])
                            input_qids.append(qid)
                            i += 1

                            if i == batch_size:
                                input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).input_ids.to(device)
                                gen_ids = model.generate(input_ids, max_new_tokens=64, top_p=0.95, repetition_penalty=1.0, temperature=0.7, num_return_sequences=1)
                                output = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                                for j in range(len(input_qids)):
                                    rev_queries[input_qids[j]] = output[j]
                                i = 0
                                input_texts, input_qids = [], []
                if i > 0:
                    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).input_ids.to(device)
                    gen_ids = model.generate(input_ids, max_new_tokens=64, top_p=0.95, repetition_penalty=1.0, temperature=0.7, num_return_sequences=1)
                    output = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    for j in range(len(input_qids)):
                        rev_queries[input_qids[j]] = output[j]
                self.queries = rev_queries
            elif method == 4: # [MASK] random 1 + put to distilbert-base-uncased {pseudo-doc [SEP] doc} to revise
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
                model.to(device)

                batch_size = 64
                rev_queries = {}

                print("Revising queries ... ")
                i = 0
                diffcnt = 0
                input_texts, input_qids = [], []
                for qid in tqdm(self.qrels):
                    for pid in self.qrels[qid]:
                        if self.qrels[qid][pid] == 1:
                            if not len(self.queries[qid]): continue 

                            query = self.queries[qid].split()
                            ridxs = np.random.choice(len(query), math.ceil(len(query)/10) if method == 5 else 1, replace=False)
                            for ridx in ridxs:
                                query[ridx] = "[MASK]"
                                input_qids.append(qid)
                            query = ' '.join(query)
                            input_texts.append(query + " [SEP] " + self.corpus[pid]["text"])    
                            rev_queries[qid] = query
                            i += 1

                            if i == batch_size:
                                inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                                mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
                                logits = model(**inputs).logits
                                for j, (x, y) in enumerate(zip(mask_token_index[0], mask_token_index[1])):
                                    mask_token_logits = logits[x, y, :]
                                    top_1_tokens = torch.topk(mask_token_logits, 1, dim=0).indices
                                    token = tokenizer.decode(top_1_tokens)
                                    rev_queries[input_qids[j]] = rev_queries[input_qids[j]].replace('[MASK]', token, 1) 
                                i = 0
                                input_texts, input_qids = [], []
                if i > 0:
                    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
                    logits = model(**inputs).logits
                    for j, (x, y) in enumerate(zip(mask_token_index[0], mask_token_index[1])):
                        mask_token_logits = logits[x, y, :]
                        top_1_tokens = torch.topk(mask_token_logits, 1, dim=0).indices
                        token = tokenizer.decode(top_1_tokens)
                        rev_queries[input_qids[j]] = rev_queries[input_qids[j]].replace('[MASK]', token, 1)
                self.queries = rev_queries
                with open(f"DATA/method{method}/{data_rev_dict[dataset_name]}/qgen-queries-revised.json", "w") as f1:
                    json.dump(rev_queries, f1, indent=4)
                logger.info("Revised Queries %d", diffcnt)

            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus
    
    def _load_corpus(self):
    
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }
    
    def _load_queries(self):
        
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score