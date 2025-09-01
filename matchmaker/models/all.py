from matchmaker.models.bert_dot import BERT_Dot
from transformers import AutoTokenizer

from matchmaker.models.colbert import ColBERT


#from matchmaker.models.private.qa_bert_cat import *
#from matchmaker.models.private.bert_dot_qa import *


def get_word_embedder(config):
    padding_idx = 0
    word_embedder = None
    model = config["bert_pretrained_model"]
    # padding_idx = PretrainedTransformerIndexer(model_name=model)._tokenizer.pad_token_id
    padding_idx = AutoTokenizer.from_pretrained(model).pad_token_id
    return word_embedder,padding_idx



def get_model(config,word_embedder,padding_idx):

    model_conf = config["model"]
    encoder_type = None
    
    # if model_conf == "bert_cls" or model_conf == "bert_cat": model = BERT_Cat.from_config(config)
    if model_conf == "bert_tower" or model_conf == "bert_dot": 
        model = BERT_Dot.from_config(config)
    
    elif model_conf == "ColBERT":
        model = ColBERT.from_config(config)


    return model, encoder_type