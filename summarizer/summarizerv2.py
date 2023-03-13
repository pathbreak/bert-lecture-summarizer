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
 
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer
import numpy as np
from numpy import ndarray
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

    
    summary_sentences = summ.summarize(lecture_content, args.ratio)
    
    if args.nojoin:
        for s in summary_sentences:
            print(textwrap.fill(s.strip(), 80))
    else:
        summary: str = ' '.join(summary_sentences).strip()
        print(textwrap.fill(summary, 80))




class SummarizerV2(object):
    def summarize(self, lecture_content, ratio):
        initial_sentences: List[str] = PreProcessor().process_content_sentences(lecture_content)
        num_sent = len(initial_sentences)
        if  num_sent == 0:
            raise RuntimeError("No viable sentences found. Consider adding larger lectures.")
        L.info(f'Content length:{num_sent}; ratio={ratio}; summary={ratio if ratio >= 1 else num_sent * ratio}')

        model = SingleModelProcessor()
        sentences = model.run_clusters(initial_sentences, ratio)

        #result: str = ' '.join(sentences).strip()
        #return result
        return sentences
        
    
class SingleModelProcessor(object):

    def __init__(self, model='bert', model_size='large', use_hidden=True):
        self.emb_model = BertLegacyEmbeddingModel(model, {
            'model_type' : 'bert',
            'size' : 'large',
            'use_hidden' : use_hidden
        })


    def run_clusters(self, sentences: List[str], ratio=0.2) -> List[str]:

        L.info('Creating embeddings matrix')
        #embeddings = self.model.create_matrix(sentences, self.use_hidden)
        embeddings = self.emb_model.embeddings(sentences)
        L.info(f'Embeddings matrix shape: {embeddings.shape}')

        L.info('Clustering embeddings')
        centroid_sent_idxes = ClusterFeatures(embeddings).cluster(ratio)
        L.info(f'centroid_sent_idxes: len={len(centroid_sent_idxes)}: {centroid_sent_idxes}')
        
        # Not clear why this hardcoded logic exists to always include the
        # first sentence in the summary. But its consequence is that there's
        # always one sentence more than what user requests.
        # if hidden_args[0] != 0:
        #    hidden_args.insert(0,0)

        return [sentences[i] for i in centroid_sent_idxes]



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
        'openApi': GPT2Model
    }

    token_handler = {
        'openApi': GPT2Tokenizer,
        'bert': BertTokenizer
    }

    size_handler = {
        'base': {
            'bert': 'bert-base-uncased',
            'openApi': 'gpt2'
        },
        'large': {
            'bert': 'bert-large-uncased',
            'openApi': 'gpt2'
        }
    }

    vector_handler = {
        'base': {
            'bert': 768,
            'openApi': 768
        },
        'large': {
            'bert': 1024,
            'openApi': 768
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
        if self.model_type == 'openApi':
            pooled = hidden_states.mean(dim=1)
        if squeeze:
            return pooled.detach().numpy().squeeze()
        return pooled

    def tokenize_input(self, text) -> torch.tensor:
        if self.model_type == 'openApi':
            indexed_tokens = self.tokenizer.encode(text)
        else:
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens])

    
class ClusterFeatures(object):

    def __init__(self, features, algorithm='kmeans', pca_k=None):
        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
        else:
            self.features = features
        self.algorithm = algorithm
        self.pca_k = pca_k

    def cluster(self, ratio=0.1):
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
        return [c for c in sentences if len(c) > 75 and not c.lower().startswith('but') and
                not c.lower().startswith('and')
                and not c.lower().__contains__('quiz') and
                not c.lower().startswith('or')]



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
    summ_cmd.add_argument('ratio', metavar='RATIO', type=float, 
                          help='Ratio of summary. <1 for ratio, >=1 for number of sentences')
    summ_cmd.add_argument('--nojoin', action='store_true', 
                          help='Print summary as list of sentences instead of joined paragraph.')
    #X_cmd.add_argument('--flag-def-false', dest='flag_def_false', action='store_true', help='Boolean flag default false')
    #X_cmd.add_argument('--flag-def-true', dest='flag_def_true' action='store_false', help='Boolean flag default true')

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

