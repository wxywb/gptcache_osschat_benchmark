import time
import sys
import os
import json
import ipdb

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation
from gptcache.similarity_evaluation.kreciprocal import KReciprocalEvaluation

class WrapEvaluation(SearchDistanceEvaluation):
    def evaluation(self, src_dict, cache_dict, **kwargs):
        return super().evaluation(src_dict, cache_dict, **kwargs)

    def range(self):
        return super().range()

def config0(embedding_f, dm):
    cache.init(
        embedding_func=embedding_f.to_embeddings,
        data_manager=dm, similarity_evaluation= WrapEvaluation(),
        config=Config(similarity_threshold=0.95),
    )

def config1(embedding_f, dm):
    cache.init(
        embedding_func=embedding_f.to_embeddings,
        data_manager=dm,
        similarity_evaluation= WrapEvaluation(),
        config=Config(similarity_threshold=0.5),
    )
    
def config2(embedding_f, dm):
    cache.init(
        embedding_func=embedding_f.to_embeddings,
        data_manager=dm,
        similarity_evaluation=OnnxModelEvaluation(),
        config=Config(similarity_threshold=0.5),
    )

def config3(embedding_f, dm):
    cache.init(
        embedding_func=embedding_f.to_embeddings,
        data_manager=dm,
        similarity_evaluation=KReciprocalEvaluation(dm.v, top_k = 2),
        config=Config(similarity_threshold=0.5),
    )

def run():
    with open('data/cluster_user_questions.json', 'r') as f:
        data = f.read()
    questions = json.loads(data)
    embedding_onnx = EmbeddingOnnx()
    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=100000)

    config3(embedding_onnx, data_manager) 

    queries = []
    cached_ques = []

    for idx, question in enumerate(questions):
        queries.append(question[0]) 
        for qi in range(1, len(question)):
            cached_ques.append({'q': question[qi], 'a': '{}_{}'.format(idx, qi)})

    cquestions = [q['q'] for q in cached_ques]
    canswers =   [q['a'] for q in cached_ques]

    if not has_data:
        cache.import_data(questions=cquestions, answers=canswers)
    #for cq, ca in zip(cquestions, canswers):
    #    cache.import_data(questions=[cq], answers=[ca])

    failed = 0 
    tp = 0  
    fp = 0
    for i, query in enumerate(queries):
        mock_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=mock_messages,
            )
            res_text = openai.get_message_from_openai_answer(res)
            cached_idx = int(res_text.split('_')[0])
            if i == cached_idx:
                tp = tp + 1
            else:
                fp = fp + 1
        except Exception as e:
            failed = failed + 1

        print('tp:', tp, 'fp:', fp, 'failed:', failed)

if __name__ == '__main__':
    run()
