import gpl
import torch, gc
import argparse

gc.collect()
torch.cuda.empty_cache()

'''
methods

0 : default
1 : generate pseudo-document w/ flan-t5-xl and task-specific prompt (auto_model)
2 : pseudo-document is revised w/ flan-t5-xl based on doc (data_loader)
3 : erase pseudo-document based on confidence (auto_model)
4 : [MASK] random 1 + put to distilbert-base-uncased {pseudo-doc [SEP] doc} to revise (data_loader)
5 : [MASK] pseudo-document based on confidence + put to distilbert-base-uncased {pseudo-doc [SEP] doc} to revise (auto_model)
6 : concatenate v3 with generated query (auto_model)
7 : concatenate v5 with generated query (auto_model)

-- not implemented --
8 : revise v3 with flan-t5-xl (auto_model)
9 : generate pseudo-document w/ flan-t5-xl and one-shot example
10 : [MASK] pseudo-document based on confidence + put to decoder {prompt, pseudo-doc, doc} to revise (X)
11 : [MASK] pseudo-document based on confidence + retrieve relevant docs to revise (X)
12 : generate queries in style of msmarco queries (distribution, few-shot, etc..)
'''

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "--dataset",
                required=True,
                type=str,       
                help="arguana, nfcorpus, scifact"
        )
        parser.add_arguemnt(
                "--method",
                type=int,
                default=0
        )
        args = parser.parse_args()

        data_dict = {'arguana':0, 'fiqa':1, 'nfcorpus':2, 'touche':3, 'scifact':4, 'fever':5, 'covid':6}
        gpl.train(
                method=args.method, # modify
                dataset_name=data_dict[args.dataset],
                path_to_generated_data=f"DATA/method{args.method}/{args.dataset}", 
                base_ckpt=f"GPL/msmarco-distilbert-margin-mse",
                gpl_score_function="dot",
                batch_size_gpl=32,
                gpl_steps=70000,
                new_size=-1,
                queries_per_passage=-1, 
                output_dir=f"checkpoints/output/method{args.method}/{args.dataset}",
                evaluation_data=f"dataset/{args.dataset}", # place dataset under dataset directory
                evaluation_output=f"checkpoints/eval/method{args.method}/{args.dataset}",
                generator="models/query-gen-msmarco-t5-base-v1", # generator="models/flan-t5-xl",
                retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
                retriever_score_functions=["cos_sim", "cos_sim"],
                cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
                qgen_prefix="qgen",
                do_evaluation=True,
                batch_size_generation=8,
                max_seq_length=200,
        )
