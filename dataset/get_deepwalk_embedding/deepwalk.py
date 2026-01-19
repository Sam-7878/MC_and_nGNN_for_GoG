import random
import numpy as np
from gensim.models import Word2Vec

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, embedding_dim, seed=42):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.seed = seed  # Seed 추가

    def random_walk(self, start_node):
        walk = [start_node]
        # Python의 random 모듈은 외부에서 seed를 고정하면 결정론적으로 작동함
        while len(walk) < self.walk_length:
            cur = walk[-1]
            # NetworkX neighbors는 순서가 보장되지 않을 수 있으므로 정렬 후 선택하거나
            # 그래프 생성 시점에 순서를 고정해야 완벽하지만, 여기서는 list 변환으로 최소한의 조치
            neighbors = sorted(list(self.graph.neighbors(cur)))
            if neighbors:
                walk.append(random.choice(neighbors))
            else:
                break
        return walk

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        
        # 워크 생성의 무작위성 제어
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        for _ in range(self.num_walks):
            random.shuffle(nodes) # 노드 순서 섞기 제어
            for node in nodes:
                walks.append(self.random_walk(node))
        return walks

    def train(self, walks):
        # Gensim Word2Vec에도 seed 전달 및 workers=1로 설정하여 결정론적 동작 유도
        # (병렬 처리 시 순서 문제로 미세한 차이가 발생할 수 있어 연구용으론 workers=1 권장)
        walks = [list(map(str, walk)) for walk in walks]  
        model = Word2Vec(
            sentences=walks, 
            vector_size=self.embedding_dim, 
            window=5, 
            min_count=0, 
            sg=1, 
            workers=1,  # 재현성을 위해 1로 설정 (속도는 느려짐). 속도가 중요하면 4로 하되 seed 고정
            seed=self.seed
        )
        return model

    def get_embeddings(self, model):
        embeddings = {}
        for node in self.graph.nodes():
            try:
                embeddings[node] = model.wv[str(node)]
            except KeyError:
                # 만약 학습되지 않은 노드가 있다면 0으로 채움 (안전장치)
                embeddings[node] = np.zeros(self.embedding_dim)
        return embeddings

