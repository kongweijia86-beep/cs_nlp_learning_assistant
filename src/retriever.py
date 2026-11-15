import numpy as np
class Retriever:
    """基于embedding相似度从chunks里找最相关的文本块"""
    def __init__(self,chunks,embeddings):
        self.chunks = chunks
        self.embeddings = embeddings
    def search(self,query_embedding,top_k=3):
        """单查询：返回 list[dict]，每个 dict 包含：
          - text:   chunk 文本
          - page:   页码
          - source: 来源 PDF
          - score:  余弦相似度（已归一化，等价于点积）
          - raw_score: 同 score，保留给 RRF 排查用
          - idx:    chunk 在 self.chunks 里的下标，供去重使用"""
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            idx = int(idx)
            s = float(scores[idx])
            results.append({
                'text':self.chunks[idx]['text'],
                'page':self.chunks[idx]['page'],
                'source':self.chunks[idx]['source'],
                'score':s,
                'raw_score':s,
                'idx':idx
            })
        return results
    def search_multi_query(
            self,
            query_embeddings,
            top_k=5,
            per_query_k=10,#每个子查询各取前多少个
            fusion = 'rrf', #融合策略
            rrf_k = 60 #RRF常数
                ):
        """多查询稠密检索"""
        per_query_results = []
        for q_emb in query_embeddings:
            res = self.search(q_emb,top_k=per_query_k)
            per_query_results.append(res)
        if fusion == 'rrf':
            return self._rrf_fusion(per_query_results,top_k,rrf_k)
        elif fusion == 'max':
            return self._max_fusion(per_query_results,top_k)
        else:
            raise ValueError(f"unknown fusion:{fusion}")
        #chunk 级去重：同一个 chunk 被多个查询命中 → 累加（Retriever 的 RRF）,
        #多个查询（子查询）各自检索 chunk，用 RRF 融合成 chunk 列表。
    def _rrf_fusion(self,per_query_results,top_k,rrf_k):
        """rrf:每个chunk得分 = sum(1/(rrf_k+rank)),
        用 idx 去重（同一 chunk 被多个子查询命中，RRF 分累加）,
        raw_score 记录该 chunk 见过的最高原始相似度（排查用）"""
        scores = {} #记录每个chunk的累计RRF分数
        chunk_map = {} #该chunk的原始dict
        best_raw = {} #该chunk见过的最高原始相似度
        for results in per_query_results:
            for rank,r in enumerate(results):
                idx = r['idx']
                scores[idx] = scores.get(idx,0.0) + 1.0/(rrf_k+rank+1)
                chunk_map[idx] = r
                best_raw[idx] = max(best_raw.get(idx,-1e9),r['raw_score'])
        ranked = sorted(scores.items(),key=lambda x:x[1],reverse = True)[:top_k]
        return [
            {
                'source':chunk_map[idx]['source'],
                'page':chunk_map[idx]['page'],
                'score':float(scores[idx]),
                'raw_score':float(best_raw[idx]), #原始相似度
                'idx':idx,
                'text':chunk_map[idx]['text']
            }
            for idx,_ in ranked
        ]
    def _max_fusion(self,per_query_results,top_k):
        """对每个chunk进行max fusion"""
        best = {}
        for results in per_query_results:
            for r in results:
                idx = r['idx']
                if idx not in best or best[idx]['score'] < r['score']:
                    best[idx] = r
        ranked = sorted(best.values(),key = lambda x:x['score'],reverse=True)[:top_k]
        return [
            {
                'source': r['source'],
                'page': r['page'],
                'score': float(r['score']),
                'raw_score': float(r['raw_score']),
                'idx': r['idx'],
                'text': r['text'],
            }
            for r in ranked
        ]

