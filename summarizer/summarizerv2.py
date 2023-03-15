# Standalone summarizer script designed for enhancements like
# using different language models, clustering algorithms, 
# new frameworks like HF transformers and spacy pipeines
# that are available now in 2023.
#
# Doesn't depend on Flask server, database, or any of the other
# components. However, its summarization logic is based on
# lecture_summarizer.py and pre-processing code from SummarizationService.py.


import argparse
import json
import sys
import os
from pprint import pprint
import logging as L
import textwrap
import bisect
 
from pytorch_pretrained_bert import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer
import torch
from sentence_transformers import SentenceTransformer
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
from numpy import ndarray
import spacy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
import nltk

from typing import List
from tqdm import tqdm
import coloredlogs

def main():
    args, parser = parse_args()
    if len(sys.argv) <= 1:
        parser.print_help()
        return 0
    
    pprint(args)
    
    log_level = getattr(args, 'log', 'DEBUG') # test command path won't have a log argument. So default to debug for tests.
    
    _setup_logging(log_level)

    fn = f'process_cmd_{args.command}'
    ret = globals()[fn](args)
    
    return ret



def process_cmd_summarize(args):
    summ = SummarizerV2()
    
    with open(args.lecture, 'r') as f:
        lecture_content = f.read()

    
    summary_sentences = summ.summarize(lecture_content, args)
    
    if args.nojoin:
        for s in summary_sentences:
            print(textwrap.fill(s.strip(), 80))
    else:
        summary: str = ' '.join(summary_sentences).strip()
        print(textwrap.fill(summary, 80))
        
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for s in summary_sentences:
                f.write(s.strip() + '\n')


def process_cmd_analyze(args):
    with open(args.lecture, 'r') as f:
        lecture_content = f.read()

    L.info('spacy start')
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(lecture_content)
    L.info('spacy end')
    
    sent_lens = []
    for s in doc.sents:
        sent_lens.append(len(s.text))
        
    arr = np.array(sent_lens)
    L.info(f'Min:{arr.min()}, Max: {arr.max()}')
    
    bin_edges = [i for i in range(0, arr.max(), args.binwidth)]
    bin_edges.append(arr.max())
    h = np.histogram(arr, bin_edges)
    print(h)



class SummarizerV2(object):
    def summarize(self, lecture_content, args):
        
        # Code based on lecture_summarizer.py > SingleModelProcessor.
        
        # Using spacy for sentence segmentation and coreference detection.
        L.info('spacy start')
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe('coreferee')
        doc = nlp(lecture_content)
        L.info('spacy end')
        
        sentences, coref_sentences = self.build_sentence_info(doc, args)
        
        #sentences: List[str] = PreProcessor().process_content_sentences(lecture_content)
        num_sent = len(sentences)
        ratio = args.ratio
        if  num_sent == 0:
            raise RuntimeError("No viable sentences found. Consider adding larger lectures.")
        
        L.info(f'Content length:{num_sent}; ratio={ratio}; summary={ratio if ratio >= 1 else num_sent * ratio}')

        L.info('Creating embeddings matrix')
        #embeddings = self.model.create_matrix(sentences, self.use_hidden)
        
        emb_model = self.create_embedding_model(args)
        
        embeddings = emb_model.embeddings(sentences)
        
        L.info(f'Embeddings matrix shape: {embeddings.shape}')
        
        
        centroid_sent_idxes = ClusterFeatures(embeddings).cluster(ratio)
        L.info(f'centroid_sent_idxes: len={len(centroid_sent_idxes)}: {centroid_sent_idxes}')
        
        # Not clear why this hardcoded logic exists to always include the
        # first sentence in the summary. But its consequence is that there's
        # always one sentence more than what user requests.
        # if hidden_args[0] != 0:
        #    hidden_args.insert(0,0)

        # If a selected sentence is part of a coreference group, include
        # previous sentences of that group too in the summary.
        if args.nocoref:
            L.info('Skipping coreference resolution')
            summary_sentences = [sentences[i] for i in centroid_sent_idxes]
        else:
            L.info('Coreference resolution')
            summary_sentence_indexes = self.include_coreferenced_sentences(centroid_sent_idxes, coref_sentences)
            summary_sentences = [sentences[i] for i in summary_sentence_indexes]
            

        return summary_sentences
    
    
    def build_sentence_info(self, doc, args):
        """Get a list of sentences and set of coreferenced sentence indexes.

        Args:
            doc (spacy.Document): The spacy document
            args (Namespace): Arguments to the tool.
            
        Returns:
            Tuple of (sentences, coref_sentences) where
                sentences -> List[str]: List of sentences in the doc
                coref_sentences -> List[Set[int]]: List of coreferencing sentence sets. Each
                    set contains indexes into `sentences`.
        """
        # If the sentences selected for summary contain coreferences, then
        # including the sentences with previous references may make the quality of
        # summary better.
        # So we identify sets of sentences that contain coreferences. To do that,
        # we have to infer the sentence indexes that contain coreferring tokens.
        # The biset.bisect_left() allows us to find those sentence indexes quickly.
        # Example: If a coref chain contains 5th and 9th token, we find their
        # corresponding sentence indexes and cache them here in coref_sentences.
        L.info('Building sentence info')
        sentence_ends = []
        skipped_sent_idxes = set()
        for sent_i, sent in enumerate(doc.sents):

            # NOTE: For some reason, spacy may include completely empty lines
            # as sentences and that trips up some torch logic while calculating
            # embeddings. Skip empty sentences.
            sent_text = sent.text.strip()
            if not sent_text:
                L.info('Skip empty sentence')
                skipped_sent_idxes.add(sent_i)
            
            if len(sent_text) < args.min_length:
                L.info('Skip short sentence')
                skipped_sent_idxes.add(sent_i)

            sentence_ends.append(sent.end)
            #print(f'{sent_i}: {sent}')

        L.info(f'Original sentences:{sent_i+1}')
        L.info(f'Skipped sentence indexes:{skipped_sent_idxes}')

        # This is a list of sets where each set is a sentence group related by references
        # to the same entity.
        coref_sentences = []
        if not args.nocoref:
            for coref_chain in doc._.coref_chains:
                sents_for_chain = set()
                for mention in coref_chain:
                    token_idx = mention[0]
                    sent_i = bisect.bisect_left(sentence_ends, token_idx)
                    #tok = doc[token_idx]
                    #print(f'{tok}: tok #{token_idx} sent #{sent_i}: {tok.sent}')
                    if sent_i not in skipped_sent_idxes:
                        L.info(f'Skipping #{sent_i} from coref chain')
                        sents_for_chain.add(sent_i)
                #print('\n\n')
                coref_sentences.append(sents_for_chain)
        
        sentences = []
        for sent_i, sent in enumerate(doc.sents):
            if sent_i not in skipped_sent_idxes:
                sentences.append(sent.text)
        L.info(f'Sentences:{len(sentences)}')
        return sentences, coref_sentences
    
    
    def create_embedding_model(self, args):
        
        if args.emb_approach == 'sbert':
            L.info(f'Loading sentence transformer {args.model}')
            emb_model = SBertEmbeddingModel({
                'model': args.model #'all-mpnet-base-v2' #all-MiniLM-L6-v2
            }) 
        
        elif args.emb_approach == 'hf':
            L.info(f'Loading HF transformer {args.model}')
            emb_model = HFTransformersEmbeddingModel({
                'model': args.model # 'facebook/bart-large' #'distilbert-base-cased'
            })
        
        elif args.emb_approach == 'bertlegacy':
            L.info(f'Loading legacy model {args.model}')
            if args.model == 'bert-base':
                model_type = 'bert'
                model_size = 'base'
                
            elif args.model == 'bert-large':
                model_type = 'bert'
                model_size = 'large'
                
            elif args.model == 'gpt2':
                model_type = 'gpt2'
                model_size = 'gpt2'
                
            
            emb_model = BertLegacyEmbeddingModel({
                'model_type' : model_type, #'bert'
                'size' : model_size, #'large',
                'use_hidden' : True
            })
        
        else:
            raise RuntimeError(f'Unknown embedding model {args.emb_model}')
        
        return emb_model
    
    
    def include_coreferenced_sentences(self, sel_sentences, coref_sentences):
        summary_sentence_indexes = []
        for sel_sent_idx in sel_sentences:
            for sents_of_chain in coref_sentences:
                if sel_sent_idx in sents_of_chain:
                    for coref_sent_i in sents_of_chain:
                        if coref_sent_i < sel_sent_idx:
                            summary_sentence_indexes.append(coref_sent_i)
            summary_sentence_indexes.append(sel_sent_idx)
            
        return summary_sentence_indexes
        

    

class EmbeddingModel(object):
    """ Base class for all embedding implementations."""

    def __init__(self, config: dict):
        self.config = config
    
    def embeddings(self, sentences: List[str]) -> ndarray:
        """
        Calculate embeddings for the given sentences.
        
        Args:
            sentences -> List[str] : List of sentences.
            
        Returns:
            A `numpy.ndarray` with shape (N, E) where N=len(sentences)
            and E is the dimension of embedding vectors produced by this
            model.
        
        """ 
        pass
    

class ClusteringModel(object):
    """ Base class for all clustering implementations."""

    def __init__(self, config: dict):
        self.config = config
    
    def cluster(self, embeddings: ndarray) -> List[int]:
        """
        Cluster the given embeddings.

        Args:
            embeddings -> ndarray: The embeddings with shape (N,E) where N=len(sentences)
            and E is the dimension of embedding vectors.
            
        Returns:
            A `List[int]` with the indexes of the embeddings that are closest
            to the cluster centers.
        """
        pass



class BertLegacyEmbeddingModel(EmbeddingModel):
    """
    Implementation copied from BertParent.py that uses the 
    legacy pytorch_pretrained_bert library.
    """

    model_handler = {
        'bert': BertModel,
        'gpt2': GPT2Model
    }

    token_handler = {
        'gpt2': GPT2Tokenizer,
        'bert': BertTokenizer
    }

    size_handler = {
        'base': {
            'bert': 'bert-base-uncased',
            'gpt2': 'gpt2'
        },
        'large': {
            'bert': 'bert-large-uncased',
            'gpt2': 'gpt2'
        }
    }

    vector_handler = {
        'base': {
            'bert': 768,
            'gpt2': 768
        },
        'large': {
            'bert': 1024,
            'gpt2': 768
        }
    }


    def __init__(self, config: dict):
        super().__init__(config)
        
        model_type: str = self.config['model_type'] 
        self.model_type = model_type
        size: str = self.config['size']
        self.use_hidden: bool = self.config['use_hidden']

        self.model = self.model_handler[model_type].from_pretrained(self.size_handler[size][model_type])
        self.tokenizer = self.token_handler[model_type].from_pretrained(self.size_handler[size][model_type])
        self.vector_size = self.vector_handler[size][model_type]
        
        self.model.eval()        

    
    def embeddings(self, sentences: List[str]) -> ndarray:
        return self.create_matrix(sentences, self.use_hidden)

    def create_matrix(self, content: List[str], use_hidden=False) -> ndarray:
        embeddings = np.zeros((len(content), self.vector_size))
        for i, t in tqdm(enumerate(content)):
            embeddings[i] = self.extract_embedding(t, use_hidden).data.numpy()
        return embeddings

    def extract_embedding(self, text: str, use_hidden=True, squeeze=False) -> ndarray:
        tokens_tensor = self.tokenize_input(text)
        hidden_states, pooled = self.model(tokens_tensor)
        if use_hidden:
            pooled = hidden_states[-2].mean(dim=1)
        if self.model_type == 'gpt2':
            pooled = hidden_states.mean(dim=1)
        if squeeze:
            return pooled.detach().numpy().squeeze()
        return pooled

    def tokenize_input(self, text) -> torch.tensor:
        if self.model_type == 'gpt2':
            indexed_tokens = self.tokenizer.encode(text)
        else:
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens])

    
class SBertEmbeddingModel(EmbeddingModel):
    """ Embedding implementations using sentence-transformers.
    
    Models can be its own official ones: https://www.sbert.net/docs/pretrained_models.html
    Or the ones from HF: https://huggingface.co/models?library=sentence-transformers&sort=downloads
    """

    def __init__(self, config: dict):
        super().__init__(config)
    
    
    def embeddings(self, sentences: List[str]) -> ndarray:
        """
        Calculate embeddings for the given sentences.
        
        Args:
            sentences -> List[str] : List of sentences.
            
        Returns:
            A `numpy.ndarray` with shape (N, E) where N=len(sentences)
            and E is the dimension of embedding vectors produced by this
            model.
        
        """
        
        # encode() returns an ndarray.
        emb_model = SentenceTransformer(self.config['model'])

        embeddings = emb_model.encode(sentences)

        return embeddings

        

class HFTransformersEmbeddingModel(EmbeddingModel):
    
    """ Embedding implementations using HF transformers.
    

    https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2#usage-huggingface-transformers.
    
    We can skip all that by simply using sentence-transformers.
    """

    def __init__(self, config: dict):
        super().__init__(config)
    
    def embeddings(self, sentences: List[str]) -> ndarray:
        """
        Calculate embeddings for the given sentences.
        
        Args:
            sentences -> List[str] : List of sentences.
            
        Returns:
            A `numpy.ndarray` with shape (N, E) where N=len(sentences)
            and E is the dimension of embedding vectors produced by this
            model.
        
        """
        
        # This returns a List[tensor] where each tensor is of shape 
        #   (1, num_tokens_in_sentence, vector_dimensions).
        #
        # The problem with raw HF transformers is that it returns contextualized 
        # **word embeddings**. To convert them to sentence embeddings, stuff like 
        # mean pooling is necessary as explained in 
        # See https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2#usage-huggingface-transformers.
        
        # The features returned by pipeline() are exactly the same as those returned
        # by this sequence:
        #       inputs = AutoTokenizer.from_pretrained(...)("Hello, my dog is cute", return_tensors="pt")
        #       model = AutoModel.from_pretrained(...)(**inputs)
        #       embeddings = outputs.last_hidden_state # or outputs[0]
        #
        # However, we also need the attention_mask and pipeline doesn't retain it.
        # So run the model raw instead of through a pipeline, as shown in
        #   https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2#usage-huggingface-transformers
        #
        # transformer_pipeline = transformers.pipeline(
        #    'feature-extraction', 
        #    model=self.config['model'])
        #embeddings = transformer_pipeline(sentences, return_tensors=True)
        
        model_name = self.config['model']
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Perform mean pooling
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings

        attention_mask = encoded_input['attention_mask']

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sentence_embeddings =  (torch.sum(token_embeddings * input_mask_expanded, 1) / 
                                torch.clamp(input_mask_expanded.sum(1), min=1e-9))
        
        L.info(f'Embeddings: {sentence_embeddings.shape}')
        
        embeddings_ndarray = sentence_embeddings.detach().numpy()

        return embeddings_ndarray



    
class ClusterFeatures(object):

    def __init__(self, features, algorithm='kmeans', pca_k=None):
        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
        else:
            self.features = features
        self.algorithm = algorithm
        self.pca_k = pca_k

    def cluster(self, ratio=0.1):
        L.info(f'Clustering embeddings using {self.algorithm}')

        # If ratio < 1, treat it as a ratio. If it's >=1, treat it as number of sentences.
        #k = 1 if ratio * len(self.features) < 1 else int(len(self.features) * ratio)
        if ratio >= 1:
            k = int(ratio)
        else:
            k = 1 if ratio * len(self.features) < 1 else int(len(self.features) * ratio)

        L.info(f'Number of centroids: {k}. Length of features matrix:{len(self.features)}')
        
        model = self.__get_model(k).fit(self.features)
        centroids = self.__get_centroids(model)
        cluster_args = self.__find_closest_args(centroids)
        sorted_values = sorted(cluster_args.values())
        return sorted_values

    def __get_model(self, k):
        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k)
        if self.algorithm == 'affinity':
            return AffinityPropagation()
        return KMeans(n_clusters=k)

    def __get_centroids(self, model):
        if self.algorithm == 'gmm':
            return model.means_
        return model.cluster_centers_

    def __find_closest_args(self, centroids):
        centroid_min = 1e7
        cur_arg = -1
        args = {}
        used_idx = []
        for j, centroid in enumerate(centroids):
            for i, feature in enumerate(self.features):
                value = np.sum(np.abs(feature - centroid))
                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min= value
            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e7
            cur_arg = -1
        return args



class PreProcessor(object):
    def process_content_sentences(self, body: str) -> List[str]:
        sentences = nltk.tokenize.sent_tokenize(body)
        return sentences
        sel_sentences = []
        for s in sentences:
            if len(s) > 75 and \
                not s.lower().startswith('but') and \
                not s.lower().startswith('and') and \
                not s.lower().__contains__('quiz') and \
                not s.lower().startswith('or'):
                
                sel_sentences.append(s)
            else:
                L.info(f'Remove sentence:{s}')
        return sel_sentences



def parse_args():
    argp = argparse.ArgumentParser(description='Lecture Summarizer')
    
    # Common args https://stackoverflow.com/questions/7498595/python-argparse-add-argument-to-multiple-subparsers
    
    #argp.add_argument(
    #    '--log', 
    #    choices=['NONE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
    #    default='NONE', 
    #    help='Set logging level')
    
    # Parent subparser for common args. Note `add_help=False` and creation via `argparse.`
    # If using this approach, pass parents=[common_args] to the subparsers to inherit
    # args from this root parser.
    common_args = argparse.ArgumentParser(add_help=False)
    common_args_grp = common_args.add_argument_group('Common arguments')
    common_args_grp.add_argument(
        '--log', 
        type=str.upper, # transformer to make it case-insensitive
        choices=['NONE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
        default='ERROR', 
        help="Set logging level (case doesn't matter)")
    
    subp = argp.add_subparsers(title='command', dest='command', metavar='COMMAND')

    summ_cmd = subp.add_parser('summarize', parents=[common_args], description='Summarize', help='Summarize')
    summ_cmd.add_argument('lecture', metavar='LECTURE-FILE', help='The lecture text file')
    summ_cmd.add_argument('-l', metavar='MIN-LENGTH', dest='min_length', type=int, default=75, 
                          help='Minimum length for sentences.')
    summ_cmd.add_argument('-o', metavar='OUTPUT-FILE', dest='output_file', help='Name of summary output file.')
    
    summ_cmd.add_argument('emb_approach', metavar='APPROACH', help='sbert|hf|bertlegacy')
    summ_cmd.add_argument('model', metavar='MODEL-NAME', 
                          help='Name of SBERT, HF, or legacy (bert-large/bert-base/gpt2) model')
    
    summ_cmd.add_argument('ratio', metavar='RATIO', type=float, 
                          help='Ratio of summary. <1 for ratio, >=1 for number of sentences')

    summ_cmd.add_argument('--nojoin', action='store_true', 
                          help='Print summary as list of sentences instead of joined paragraph.')
    summ_cmd.add_argument('--nocoref', action='store_true', 
                          help='Disable coreference resolution.')
    #X_cmd.add_argument('--flag-def-false', dest='flag_def_false', action='store_true', help='Boolean flag default false')
    #X_cmd.add_argument('--flag-def-true', dest='flag_def_true' action='store_false', help='Boolean flag default true')

    analyze_cmd = subp.add_parser('analyze', parents=[common_args], description='Analyze sentences', 
                                       help='Analyze sentences')
    analyze_cmd.add_argument('lecture', metavar='LECTURE-FILE', help='The lecture text file')
    analyze_cmd.add_argument('binwidth', metavar='BINWIDTH', type=int, default=20, help='Bin width in num chars for histogram')

    args = argp.parse_args()
    
    return args, argp


def _setup_logging(log_level):
    if log_level == 'NONE':
        logging.disable(logging.CRITICAL)
    else:
        coloredlogs.install(level=log_level, 
            fmt='%(asctime)s %(levelname)s %(message)s')
        
    #logging.debug('debug')
    #logging.info('info')
    #logging.warning('warning')
    #logging.error('error')
    #logging.critical('critical')
    #logging.exception('exception')
    
    
if __name__ == '__main__':
    main()

