import os
import json
import time
from pathlib import Path
import faulthandler
faulthandler.dump_traceback_later(300,repeat=True)
import numpy as np
import torch
from src.rag import RAG
BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR/'eval'/'questions.json'

def dcg(relevances,k):
    """计算 DCG（折损累计增益）：给排在前面的结果打高分，越靠前权重越大。"""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2,relevances.size+2))
    return np.sum(relevances/discounts)

def ndcg_at_k(relevances,k):
    """计算 NDCG = DCG / IDCG，结果在 [0,1]，越接近 1 排序越完美。"""
    ideal_relevances = sorted(relevances,reverse=True)
    dcg_val = dcg(relevances,k)
    idcg_val = dcg(ideal_relevances,k)
    return dcg_val/idcg_val if idcg_val > 0 else 0.0

def resolve_relevant_pages(item):
    """从题目里提取标准答案所在页码，统一成 [{'source':..., 'page':...}] 格式。
       """
    relevant_pages = item.get('relevant_pages')
    if relevant_pages:
        return [{'source':rp['source'],'page':rp['page']} for rp in relevant_pages]
    page_hints = item.get('page_hints')
    if page_hints:
        source = item.get('source',"Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf")
        return [{'source':source,'page':p} for p in page_hints]
    page_hint = item.get('page_hint')
    if page_hint is not None:
        source = item.get('source',"Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf")
        return [{'source':source,'page':page_hint}]
    return []

def _rank_metrics(results,relevant_pages,k):
    """在给定排序结果上，计算 MRR@k 和 NDCG@k，返回 (mrr, ndcg)。"""
    mrr = 0.0
    relevances = []
    for i,r in enumerate(results[:k]):
        is_rel = (r['source'],r['page']) in relevant_pages
        relevances.append(1 if is_rel else 0)
        if is_rel and mrr == 0.0:
            mrr = 1.0/(i + 1)
    return mrr, ndcg_at_k(relevances, k=k)

def run_ablation():
    print("环境诊断", flush=True)
    print(f"  CUDA 可用: {torch.cuda.is_available()}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("  警告: 未检测到 GPU，Reranker 将在 CPU 上推理，可能非常慢", flush=True)
    print(f"  HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE')}", flush=True)
    print(f"  HF_TOKEN 已设置 = {bool(os.environ.get('HF_TOKEN'))}", flush=True)
    print('正在初始化RAG',flush=True)
    t0 = time.time()
    rag = RAG(str(BASE_DIR/'data'/'raw'),use_reranker=True)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"RAG 初始化完成，耗时 {time.time() - t0:.1f} 秒", flush=True)
    model_name = getattr(getattr(rag, "reranker", None), "model_name", "unknown")
    print(f"Reranker 模型: {model_name}", flush=True)

    with open(QUESTIONS_FILE,'r',encoding='utf-8') as f:
        data = json.load(f)
    print(f"共加载{len(data)}道题",flush=True)
    # 统一截断长度 / 候选池大小，保证各组指标可比
    K = 20
    K_TOP = 5 #额外做一组top5排序质量对比
    metrics = {
        # 三档：Dense → Hybrid → Hybrid+Rerank
        'recall_dense':[],"recall_hybrid":[],
        'mrr_dense':[],'mrr_hybrid':[],'mrr_rerank':[],
        'ndcg_dense':[],'ndcg_hybrid':[],'ndcg_rerank':[],
        # top-5 排序质量对比
        'ndcg_d5':[],'ndcg_h5':[],'ndcg_r5':[],
        # 逐题增量（rerank - hybrid）
        "mrr_delta": [], "ndcg_delta": [],
        #n_valid表示有效题目数，用来统计真正参与了指标计算，且没有被跳过的题目数量
        "n_valid": 0,
    }
    for idx,item in enumerate(data):
        q = item['question']
        rels = resolve_relevant_pages(item)
        relevant_pages = set((rp['source'],rp['page']) for rp in rels)
        if not relevant_pages:
            print(f"\n[{idx}] 跳过（无标准答案）: {q[:40]}...", flush=True)
            continue
        metrics['n_valid'] += 1
        t_item = time.time()
        print(f"\n[{idx}/{len(data)}] {q[:40]}...", flush=True)
        print('encode...',flush=True)
        t = time.time()
        query_emb = rag.embedding_model.encode(q)[0]
        if device == "cuda": torch.cuda.synchronize()
        print(f"    encode 耗时 {time.time() - t:.2f}s", flush=True)
        #对照组1：纯向量搜索
        print("dense...",flush=True)
        t = time.time()
        candidates_dense = rag.retriever.search(query_emb,top_k=K)
        if device == "cuda": torch.cuda.synchronize()
        print(f'dense耗时:{time.time()-t:.2f}s',flush=True)
        retrieved_dense_set = set((r['source'],r['page']) for r in candidates_dense)
        recall_dense = len(retrieved_dense_set & relevant_pages) / len(relevant_pages)
        metrics['recall_dense'].append(recall_dense)
        mrr_d,ndcg_d = _rank_metrics(candidates_dense,relevant_pages,K)
        metrics['mrr_dense'].append(mrr_d)
        metrics['ndcg_dense'].append(ndcg_d)
        _,ndcg_d5 = _rank_metrics(candidates_dense,relevant_pages,K_TOP)
        metrics['ndcg_d5'].append(ndcg_d5)
        #对照组2：混合搜索（Dense+BM25+RRF)
        print('bm25...',flush=True)
        t = time.time()
        bm25_results = rag.bm25_retriever.search(q,top_k=K)
        if device == "cuda": torch.cuda.synchronize()
        print(f"bm25耗时{time.time()-t:.2f}s",flush=True)
        print('rrf...',flush=True)
        candidates_hybrid = rag._rrf_fuse(candidates_dense, bm25_results)[:K]
        retrieved_hybrid_set = set((r['source'],r['page']) for r in candidates_hybrid)
        recall_hybrid = len(retrieved_hybrid_set & relevant_pages)/len(relevant_pages)
        metrics['recall_hybrid'].append(recall_hybrid)
        mrr_h,ndcg_h = _rank_metrics(candidates_hybrid,relevant_pages,K)
        metrics['mrr_hybrid'].append(mrr_h)
        metrics['ndcg_hybrid'].append(ndcg_h)
        #实验组：对【同一批 K 个候选】重排，截断同样为 K
        print('rerank',flush=True)
        t = time.time()
        candidates_rerank =rag.reranker.rerank(q,candidates_hybrid,top_k=K)
        if device == "cuda": torch.cuda.synchronize()
        print(f"rerank耗时:{time.time()-t:.2f}s",flush=True)
        mrr_r,ndcg_r = _rank_metrics(candidates_rerank,relevant_pages,K)
        metrics['mrr_rerank'].append(mrr_r)
        metrics['ndcg_rerank'].append(ndcg_r)
        # 逐题增量，用于统计"多少题变好/变差/持平"
        metrics['mrr_delta'].append(mrr_r-mrr_h)
        metrics['ndcg_delta'].append(ndcg_r-ndcg_h)
        _, ndcg_h5 = _rank_metrics(candidates_hybrid, relevant_pages, K_TOP)
        _, ndcg_r5 = _rank_metrics(candidates_rerank, relevant_pages, K_TOP)
        metrics["ndcg_h5"].append(ndcg_h5)
        metrics["ndcg_r5"].append(ndcg_r5)
        if device == "cuda": torch.cuda.synchronize()
        print(f"  done  (本题耗时 {time.time() - t_item:.2f}s)", flush=True)

    n = metrics["n_valid"]
    if n:
        for key in ["mrr_dense", "mrr_hybrid", "mrr_rerank",
                    "ndcg_dense", "ndcg_hybrid", "ndcg_rerank",
                    "ndcg_d5", "ndcg_h5", "ndcg_r5"]:
            assert len(metrics[key]) == n, f"指标长度不一致: {key}={len(metrics[key])}, n={n}"

    def mean(x):
        return float(np.mean(x)) if x else 0.0

    mrr_delta = np.asarray(metrics["mrr_delta"])
    ndcg_delta = np.asarray(metrics["ndcg_delta"])

    print("\n================= 消融实验最终结果 =================", flush=True)
    print(f"Reranker 模型: {model_name}", flush=True)
    print(f"有效题目数: {n}", flush=True)
    print(f"0. 纯向量 MRR (MRR@20):             {mean(metrics['mrr_dense']):.4f}", flush=True)
    print(f"0. 纯向量 NDCG (NDCG@20):           {mean(metrics['ndcg_dense']):.4f}", flush=True)
    print(f"1. 纯向量召回率 (Recall@20):        {mean(metrics['recall_dense']):.4f}", flush=True)
    print(f"2. 混合检索召回率 (Recall@20):      {mean(metrics['recall_hybrid']):.4f}  <-- 验证 RRF 价值",
          flush=True)
    print(f"3. 混合检索 MRR (MRR@20):           {mean(metrics['mrr_hybrid']):.4f}", flush=True)
    print(f"4. 混合检索 NDCG (NDCG@20):         {mean(metrics['ndcg_hybrid']):.4f}", flush=True)
    print(f"5. 混合+精排 MRR (MRR@20):          {mean(metrics['mrr_rerank']):.4f}  <-- 验证 Reranker 价值",
          flush=True)
    print(f"6. 混合+精排 NDCG (NDCG@20):        {mean(metrics['ndcg_rerank']):.4f}", flush=True)

    print("\n--------------- 同截断长度下的公平对比 ----------------", flush=True)
    print(f"RRF 带来的 Recall@20 提升:  {mean(metrics['recall_hybrid']) - mean(metrics['recall_dense']):+.4f}",
          flush=True)
    print(f"Reranker 带来的 MRR@20 提升: {mean(metrics['mrr_rerank']) - mean(metrics['mrr_hybrid']):+.4f}",
          flush=True)
    print(f"Reranker 带来的 NDCG@20 提升:{mean(metrics['ndcg_rerank']) - mean(metrics['ndcg_hybrid']):+.4f}",
          flush=True)
    print(f"精排 vs 纯向量 MRR@20 提升:  {mean(metrics['mrr_rerank']) - mean(metrics['mrr_dense']):+.4f}",
          flush=True)
    print(f"精排 vs 纯向量 NDCG@20 提升: {mean(metrics['ndcg_rerank']) - mean(metrics['ndcg_dense']):+.4f}",
          flush=True)

    print("\n--------------- 逐题增量统计（rerank - hybrid）----------------", flush=True)
    for name, delta in [("MRR@20", mrr_delta), ("NDCG@20", ndcg_delta)]:
        better = int(np.sum(delta > 1e-9))
        worse = int(np.sum(delta < -1e-9))
        same = n - better - worse
        if n:
            print(f"{name}: 变好 {better} 题 | 变差 {worse} 题 | 持平 {same} 题 "
                  f"(提升题占比 {better / n:.1%})", flush=True)
        else:
            print(f"{name}: 无有效题目", flush=True)

    print("\n--------------- Top-5 排序质量专项对比 ----------------", flush=True)
    print(f"纯向量 NDCG@5:     {mean(metrics['ndcg_d5']):.4f}", flush=True)
    print(f"混合检索 NDCG@5:   {mean(metrics['ndcg_h5']):.4f}", flush=True)
    print(f"混合+精排 NDCG@5:  {mean(metrics['ndcg_r5']):.4f}", flush=True)
    print(f"Reranker 带来的 NDCG@5 提升: {mean(metrics['ndcg_r5']) - mean(metrics['ndcg_h5']):+.4f}", flush=True)

if __name__ == "__main__":
    run_ablation()







