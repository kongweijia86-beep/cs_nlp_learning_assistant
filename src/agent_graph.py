from typing import TypedDict,Annotated,Optional
import operator
import json
import asyncio
import aiosqlite
import numpy as np
import hashlib
from openai import AsyncOpenAI
from langchain_core.messages import RemoveMessage,HumanMessage,AIMessage,ToolMessage
from langgraph.graph import StateGraph,END,START
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from src.rag import RAG

_rag_instance = None
_redis_client = None
_async_llm_client = None

def set_redis_client(client):
    global _redis_client
    _redis_client = client

async def extract_facts_async(user_msg:str,ai_msg:str)->list[str]:
    prompt = f"""提取用户在此对话中暴露的个人信息（如技术栈、研究方向、身份、偏好）。
        只输出 JSON 格式的字符串列表，如果没有提取到信息返回 []。
        对话：User: {user_msg}\nAssistant: {ai_msg[:500]}"""
    try:
        res = await _async_llm_client.chat.completions.create(
            model = 'deepseek-chat',
            messages = [{'role':'user','content':prompt}],
            temperature = 0.0,response_format = {'type':'json_object'}
        )
        data = json.loads(res.choices[0].message.content)
        return data.get('facts',[]) if isinstance(data,dict) else data if isinstance(data,list) else []
    except Exception as e:
        print(f"memory提取失败:{e}")
        return []

async def save_memory_to_vector_store_async(thread_id:str,facts:list[str]):
    """将事实存进向量库"""
    if not _redis_client or not facts:
        return
    for fact in facts:
        try:
            vector = await asyncio.to_thread(_rag_instance.embedding_model.encode,[fact],normalize_embeddings=True)
            vector = vector[0]
            vector_str = json.dumps(vector.tolist())
            fact_hash = hashlib.md5(fact.encode()).hexdigest()
            memory_key = f"memory:{thread_id}:{fact_hash}"

            await _redis_client.hset(memory_key,mapping={'vector':vector_str,'text':fact})
            await _redis_client.sadd(f"user_memory_index:{thread_id}",memory_key)
            await _redis_client.expire(memory_key,30*24*3600)
        except Exception as e:
            print(f'Memory存向量失败:{e}')

async def retrieve_relevant_memories_async(thread_id:str,query:str,top_k:int=3)->list[str]:
    """在redis中检索相关记忆"""
    if not _redis_client:
        return []
    try:
        query_vector = await asyncio.to_thread(_rag_instance.embedding_model.encode,[query],normalize_embeddings=True)
        query_vector = query_vector[0]
        keys = await _redis_client.smembers(f"user_memory_index:{thread_id}")
        if not keys:
            return []
        scored_memories = []
        for key in keys:
            data = await _redis_client.hgetall(key)
            if data and 'vector' in data and 'text' in data:
                score = float(np.dot(query_vector,np.array(json.loads(data['vector']))))
                scored_memories.append((score,data['text']))
        scored_memories.sort(key=lambda x:x[0],reverse=True)
        return [text for score,text in scored_memories[:top_k] if score > 0.5]
    except Exception as e:
        return []

def _calc_attention_complexity(d_k: int, seq_len: int, n_heads: int = 1, n_layers: int = 1) -> dict:
    flops = (4 * (seq_len ** 2) * d_k + 2 * (seq_len ** 2)) * n_heads * n_layers
    return {'flops': flops, 'flops_readable': f"{flops:3e}"}

def _calc_attention_params(d_model:int,n_heads:int)->dict:
    return {'params':4 * (d_model*d_model+d_model)}

def _calc_ffn_params(d_model:int,expansion:int=4):
    d_ff = d_model * expansion
    return {"params": (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)}

@tool
def calculate_attention_complexity(d_k:int,seq_len:int)->str:
    """计算FLOPS"""
    return json.dumps(_calc_attention_complexity(d_k,seq_len))

@tool
def calculate_attention_params(d_model:int,n_heads:int)->str:
    """计算参数量"""
    return json.dumps(_calc_attention_params(d_model,n_heads))

@tool
def calculate_ffn_params(d_model:int,expansion:int=4)->str:
    """计算FFN参数量"""
    return json.dumps(_calc_ffn_params(d_model,expansion))

@tool
def search_textbook(query:str)->str:
    """查书"""
    return "检索工具由 Agent 执行器处理"

async def retrieve_textbook(query:str,is_complex:bool=False)->list[dict]:
    """根据问题复杂度选择检索策略。返回统一格式的 contexts。"""
    if _rag_instance is None:
        return []
    if is_complex:
        contexts = await asyncio.to_thread(
            _rag_instance.search_multi_query,
            query,
            top_k = 5,
            n_queries = 2,
            per_query_k = 10,
            tolerance = 1
        )
    else:
        emb = await asyncio.to_thread(_rag_instance.embedding_model.encode,[query],normalize_embeddings=True)
        contexts = await asyncio.to_thread(
            _rag_instance.search,
            emb[0],
           query_text = query,
           top_k = 5)
    return contexts

async def _safe_llm_call_async(prompt:str,max_tokens:int=200)->str:
    try:
        response = await _async_llm_client.chat.completions.create(
            model = 'deepseek-chat',
            messages = [{'role':'user','content':prompt}],
            max_tokens = max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ''

def _history_to_text(history,last_n:int=4)->str:
    lines = []
    for m in history[-last_n:]:
        if isinstance(m,HumanMessage):
            lines.append(f"user:{m.content}")
        elif isinstance(m,AIMessage):
            lines.append(f"assistant:{m.content}")
    return '\n'.join(lines)

class AgentState(TypedDict):
    messages:Annotated[list,operator.add]
    summary:str
    question:str
    answer:str
    contexts:list[dict]
    current_topic:str
    user_profile:list[str]
    is_complex:bool
    thread_id:str

async def build_agent(rag:RAG):
    global _rag_instance,_async_llm_client
    _rag_instance = rag
    _async_llm_client = AsyncOpenAI(api_key=rag.llm_client.api_key,base_url='https://api.deepseek.com')
    llm_with_tools = ChatOpenAI(
        model='deepseek-chat', api_key=rag.llm_client.api_key, base_url="https://api.deepseek.com", temperature=0.0
    ).bind_tools([calculate_attention_complexity,calculate_attention_params,calculate_ffn_params,search_textbook])
    router_llm = ChatOpenAI(model='deepseek-chat',api_key=rag.llm_client.api_key,base_url='https://api.deepseek.com',temperature=0.0,max_tokens=5)

    async def memory_retrieval_node(state:AgentState):
        msgs = state.get('messages',[])
        last_user_msg = next((m.content for m in reversed(msgs) if isinstance(m,HumanMessage)),'')
        if not last_user_msg:
            return {}
        thread_id = state.get('thread_id','default')
        return {'user_profile':await retrieve_relevant_memories_async(thread_id,last_user_msg,top_k=3)}

    async def triage_node(state:AgentState):
        msgs = state.get('messages',[])
        last_user_msg = next((m.content for m in reversed(msgs) if isinstance(m,HumanMessage)),'')
        if not last_user_msg:
            return {'is_complex':False}
        prompt = f"判断问题是否复杂(比较、多步骤等)，只回答 yes 或 no。\n用户问题：{last_user_msg}"
        try:
            response = await router_llm.ainvoke(prompt)
            return {'is_complex':'yes' in response.content.strip().lower()}
        except Exception:
            return {'is_complex':False}

    async def agent_node(state:AgentState):
        msgs = state.get('messages',[])
        start_idx = max(0,len(msgs) - 6)
        while start_idx > 0 and isinstance(msgs[start_idx],ToolMessage):
            start_idx -= 1
        profile_facts = state.get('user_profile',[])
        profile_hint = f"\n[背景]:{';'.join(profile_facts)}" if profile_facts else ''
        complexity = '[复杂]' if state.get('is_complex') else '[简单]'
        system_prompt = f"你是NLP助手。{profile_hint} {complexity}\n 1.算参数必须用工具。\n2.查概念必用 search_textbook。\n3.加 [Page X] 引用。"
        invoke_msgs = [HumanMessage(content=system_prompt)] + msgs[start_idx:]
        response = await llm_with_tools.ainvoke(invoke_msgs)
        return {'messages':[response]}

    async def tool_node(state:AgentState):
        last_message = state['messages'][-1]
        tool_messages = []
        current_contexts = state.get('contexts',[])
        for tool_call in last_message.tool_calls:
            t_name,args = tool_call['name'],tool_call['args']
            print(f"调用了{t_name}")
            if t_name == 'search_textbook':
                query = args.get('query','')
                is_complex = state.get('is_complex',False)
                print(
                    f"问题类型: {'复杂问题' if is_complex else '简单问题'}"
                )
                contexts = await retrieve_textbook(
                    query = query,
                    is_complex = is_complex
                )
                current_contexts = contexts
                if contexts:
                    res = '\n'.join([f"[Page{c['page']}]\n{c['text']}" for c in contexts])
                else:
                    res = '没有找到相关资料'
            elif t_name == 'calculate_attention_complexity':
                res = calculate_attention_complexity.invoke(args)
            elif t_name == 'calculate_attention_params':
                res = calculate_attention_params.invoke(args)
            elif t_name == 'calculate_ffn_params':
                res = calculate_ffn_params.invoke(args)
            else:
                res = '未知工具'
            tool_messages.append(ToolMessage(content=res,tool_call_id=tool_call['id']))
        return {'messages':tool_messages,'contexts':current_contexts}

    def should_continue(state:AgentState):
        last = state['messages'][-1]
        return 'tools' if hasattr(last,'tool_calls') and last.tool_calls else 'end'

    async def post_process_node(state:AgentState):
        last_message = state['messages'][-1]
        answer = last_message.content if hasattr(last_message,'content') else str(last_message)
        topic = state.get('current_topic','')
        if answer:
            new_topic = await _safe_llm_call_async(f"提取核心实体（1词）:\n{answer[:200]}\n实体",max_tokens=20)
            if new_topic:
                topic = new_topic
        thread_id = state.get('thread_id','default')
        msgs = state.get('messages',[])
        user_msg = next((m.content for m in reversed(msgs) if isinstance(m,HumanMessage)),'')
        ai_msg = next((m.content for m in reversed(msgs) if isinstance(m,AIMessage)),'')
        if user_msg and ai_msg:
            facts = await extract_facts_async(user_msg,ai_msg)
            if facts:
                print(f"memory提取到记忆:{facts}")
                await save_memory_to_vector_store_async(thread_id,facts)
        return {'answer':answer,'current_topic':topic}

    async def summarize_node(state:AgentState):
        msgs = state['messages']
        if len(msgs) <= 10:
            return {}
        split_idx = max(0,len(msgs)-8)
        while split_idx > 0 and isinstance(msgs[split_idx],ToolMessage):
            split_idx -= 1
        old = msgs[:split_idx]
        if not old:
            return {}
        old_text = _history_to_text(old,len(old))
        summary = await _safe_llm_call_async(f"压缩摘要:\n已有：{state.get('summary','')}\n新增:\n{old_text}",max_tokens=300)
        delete_messages = [RemoveMessage(id=m.id) for m in old if getattr(m,'id',None)]
        return {'summary':summary,'messages':delete_messages} if summary else {}

    workflow = StateGraph(AgentState)
    workflow.add_node('memory_retrieval',memory_retrieval_node)
    workflow.add_node('triage',triage_node)
    workflow.add_node('agent',agent_node)
    workflow.add_node('tools',tool_node)
    workflow.add_node('post_process',post_process_node)
    workflow.add_node('summarize',summarize_node)

    workflow.add_edge(START,'memory_retrieval')
    workflow.add_edge('memory_retrieval','triage')
    workflow.add_edge('triage','agent')
    workflow.add_conditional_edges('agent',should_continue,{'tools':'tools','end':'post_process'})
    workflow.add_edge('tools','agent')
    workflow.add_edge('post_process','summarize')
    workflow.add_edge('summarize',END)

    conn = await aiosqlite.connect('checkpoint.db')
    memory = AsyncSqliteSaver(conn)
    return workflow.compile(checkpointer=memory),conn



