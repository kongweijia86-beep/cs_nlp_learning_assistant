import json
import uuid
import hashlib
import asyncio
import traceback
from pathlib import Path
from typing import Optional,List,Dict,Any
from db import save_message_to_mysql, load_history_from_mysql, init_db_pool
import redis.asyncio as aioredis
from fastapi import FastAPI,HTTPException,UploadFile,File,Query,Request
from fastapi.responses import StreamingResponse,FileResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.rag import RAG
import src.agent_graph as agent_graph

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR/'data'/'raw'
STATIC_DIR = BASE_DIR/'static'

r = None
USE_REDIS = True
app = FastAPI(title='NLP学习助手',version='4.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
STATIC_DIR.mkdir(exist_ok=True)
app.mount('/static',StaticFiles(directory=str(STATIC_DIR)),name='static')

rag_instance:Optional[RAG] = None
agent_instance = None
checkpoint_conn = None
resources_lock = asyncio.Lock()
_running_queries = {}

@app.on_event('startup')
async def startup_event():
    global r,USE_REDIS,rag_instance,agent_instance,checkpoint_conn
    print("开始初始化MySQL连接池")
    try:
        await init_db_pool()
        print('MySQL连接池OK')
    except Exception as e:
        print(f'MySQL初始化崩溃:{e}')
        traceback.print_exc()  # 打印原始报错方便排查
    print('开始初始化Redis')
    try:
        r = aioredis.Redis(host='localhost',port=6379,db=0,decode_responses=True)
        await r.ping()
        USE_REDIS = True
        agent_graph.set_redis_client(r)
        print('Redis OK')
    except Exception:
        USE_REDIS = False
        r = None
        print('Redis未连接，缓存已禁用')
    print('开始加载RAG')
    try:
        rag_instance = RAG(str(PDF_DIR))
        print('RAG OK')
    except Exception as e:
        print(f"RAG崩溃{e}")
        traceback.print_exc()
    print('开始加载Agent')
    try:
        agent_instance,checkpoint_conn = await agent_graph.build_agent(rag_instance)
        print('Agent OK')
    except Exception as e:
        print(f'Agent崩溃:{e}')
        traceback.print_exc()
    print('启动流程失败')

@app.on_event('shutdown')
async def shutdown_event():
    global checkpoint_conn
    if checkpoint_conn:
        await checkpoint_conn.close()

async def rebuild_resources():
    global rag_instance,agent_instance,checkpoint_conn
    async with resources_lock:
        print('检测到知识库变更，正在重建')
        new_rag = RAG(str(PDF_DIR))
        if checkpoint_conn:
            await checkpoint_conn.close()
        rag_instance = new_rag
        agent_instance,checkpoint_conn = await agent_graph.build_agent(rag_instance)
        print('重建完毕')

async def _safe_rebuild():
    try:
        await rebuild_resources()
    except Exception as e:
        print(f'rebuild失败:{e}')

class ChatRequest(BaseModel):
    question:str
    thread_id:Optional[str] = None
    use_cache:bool = True

class ChatResponse(BaseModel):
    thread_id:str
    answer:str
    contexts:List[Dict[str,Any]] = []
    cached:bool = False

def get_cache_key(question:str,thread_id:str='')->str:
    raw = f"{thread_id}:{question.strip().lower()}"
    return f"rag_cache:{hashlib.md5(raw.encode()).hexdigest()}"

async def save_message_to_redis(user_id:str,role:str,content:str,extra_data:dict=None):
    if not USE_REDIS:
        return
    key,message = f"chat_memory:{user_id}",{'role':role,'content':content}
    if extra_data:
        message.update(extra_data)
    try:
        await r.rpush(key,json.dumps(message,ensure_ascii=False))
        await r.ltrim(key,-20,-1)
        await r.expire(key,7*24*3600)
    except Exception:
        pass

async def load_history_from_redis(user_id:str)->List[dict]:
    if not USE_REDIS:
        return []
    try:
        raw = await r.lrange(f"chat_memory:{user_id}",0,-1)
        return [json.loads(m) for m in raw]
    except Exception:
        return []

async def cache_set(q_key:str,answer:str,contexts:list=None):
    """写入缓存"""
    if not USE_REDIS or not q_key or not answer:
        return
    try:
        if '资料中没有找到' in answer or '资料不足' in answer:
            await r.set(q_key,'__NULL__',ex=60)
        else:
            cache_data = json.dumps({
                'answer':answer,'contexts':contexts or []
            },ensure_ascii=False)
            await r.set(q_key,cache_data,ex=3600)
    except Exception:
        pass

async def zincr_hot(question:str):
    if not USE_REDIS:
        return
    try:
        await r.zincrby('hot_questions',1,question.lower().strip())
    except Exception:
        pass

async def get_history_cache_aside(thread_id:str)->list:
    if USE_REDIS:
        redis_history = await load_history_from_redis(thread_id)
        if redis_history:
            return redis_history
    print(f'Redis未命中，查MySQL:{thread_id}')
    mysql_history = await load_history_from_mysql(thread_id)
    if USE_REDIS and mysql_history:
        await r.delete(f"chat_memory:{thread_id}")
        for msg in mysql_history:
            extra = {'contexts':msg.get('contexts')} if msg.get('contexts') else None
            await save_message_to_redis(thread_id,msg['role'],msg['content'],extra)
    return mysql_history

@app.get('/',response_class=HTMLResponse)
async def index():
    f = STATIC_DIR/'index.html'
    return HTMLResponse(f.read_text(encoding='utf-8')) if f.exists() else HTMLResponse('<h1>Error</h1>',404)

@app.post('/chat', response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not agent_instance: raise HTTPException(503, "初始化失败，请检查终端日志")
    thread_id, q_key = req.thread_id or str(uuid.uuid4()), get_cache_key(req.question, thread_id) if USE_REDIS else ''
    if USE_REDIS and q_key in _running_queries:
        raise HTTPException(429,'正在生成中')
    if USE_REDIS:
        _running_queries[q_key] = True
    await zincr_hot(req.question)
    try:
        if USE_REDIS and req.use_cache:
            cached = await r.get(q_key)
            if cached == '__NULL__':
                msg = "资料不足，无法回答"
                await save_message_to_mysql(thread_id,'user',req.question)
                await save_message_to_mysql(thread_id,'assistant',msg)
                return ChatResponse(thread_id=thread_id,answer=msg,contexts=[],cached=True)
            elif cached:
                cache_data = json.loads(cached)
                cached_answer = cache_data.get('answer','')
                cached_contexts = cache_data.get('contexts',[])
                await save_message_to_mysql(thread_id,'user',req.question)
                await save_message_to_mysql(thread_id,'assistant',cached_answer,cached_contexts)
                return ChatResponse(thread_id=thread_id,answer=cached_answer,contexts=cached_contexts,cached=True)
            config = {'configurable':{'thread_id':thread_id}}
            inputs = {'messages':[HumanMessage(content=req.question)],'thread_id':thread_id}
            res_state = await agent_instance.ainvoke(inputs,config)
            ans,ctx = res_state.get('answer',''),res_state.get('contexts',[])
            await cache_set(q_key,ans)
            await save_message_to_redis(thread_id, "user", req.question)
            await save_message_to_redis(thread_id, "assistant", ans, {"contexts": ctx})
            await save_message_to_mysql(thread_id, "user", req.question)
            await save_message_to_mysql(thread_id, "assistant", ans, ctx)
            return ChatResponse(thread_id=thread_id, answer=ans, contexts=ctx)
    finally:
        if USE_REDIS:
            _running_queries.pop(q_key,None)

@app.post('/chat_stream')
async def chat_stream_endpoint(req:ChatRequest,request:Request):
    if not agent_instance:
        raise HTTPException(503,'初始化失败，请检查终端')
    thread_id = req.thread_id or str(uuid.uuid4())
    q_key = get_cache_key(req.question,thread_id) if USE_REDIS else ''
    if USE_REDIS and q_key in _running_queries:
        raise HTTPException(429,'正在生成中')
    if USE_REDIS:
        _running_queries[q_key] = True
    await zincr_hot(req.question)

    def sse(event:str,data:Any):
        return f"event:{event}\ndata:{json.dumps(data,ensure_ascii=False)}\n\n"

    async def event_generator():
        try:
            yield sse('meta',{'thread_id':thread_id})
            if USE_REDIS and req.use_cache:
                cached = await r.get(q_key)
                if cached == '__NULL__':
                    yield sse('token',{'content':'资料不足'})
                    yield sse('contexts',{'contexts':[]})
                    yield sse('done',{'cached':True})
                    await save_message_to_mysql(thread_id, "user", req.question)
                    await save_message_to_mysql(thread_id, "assistant", '资料不足')
                elif cached:
                    cached = json.loads(cached)
                    cached_answer = cached.get('answer','')
                    cached_contexts = (cached.get('contexts',[]))
                    yield sse("token",{'cached':cached_answer})
                    yield sse('contexts',{'contexts':cached_contexts})
                    yield sse('done',{'cached':True})
                    await save_message_to_mysql(thread_id,'user',req.question)
                    await save_message_to_redis(thread_id,'assistant',cached_answer,{'contexts':cached_contexts})
                    return
            config,inputs = {'configurable':{'thread_id':thread_id}},{'messages':HumanMessage(content=req.question),'thread_id':thread_id}
            full_ans,final_state = '',{}
            try:
                async for event in agent_instance.astream_events(inputs,config=config,version='v2'):
                    if await request.is_disconnected():
                        break
                    if event['event'] == 'on_chat_model_stream':
                        content = event['data']['chunk'].content
                        if content:
                            full_ans += content
                            yield sse('token',{'content':content})
                    elif event['event'] == 'on_chain_end':
                        out = event['data'].get('output',{})
                        if isinstance(out,dict) and 'answer' in out:
                            final_state.update(out)
            except Exception as e:
                yield sse('error',{'message':str(e)})
                yield sse('done',{'cached':False})
                return
            ctx = final_state.get('contexts',[]) or []
            yield sse('contexts',{'contexts':ctx})
            yield sse('done',{'cached':False})
            await cache_set(q_key,full_ans)
            await save_message_to_mysql(thread_id,'user',req.question)
            await save_message_to_mysql(thread_id,'assistant',full_ans,ctx)
        finally:
            if USE_REDIS:
                _running_queries.pop(q_key,None)
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get('/history/{thread_id}')
async def get_history(thread_id:str):
    return {'thread_id':thread_id,'history':await get_history_cache_aside(thread_id)}

@app.get('/hot_questions')
async def get_hot_questions(top:int = Query(5,ge=1,le=50)):
    if not USE_REDIS:
        return {'hot':[]}
    try:
        hot =await r.zrevrange('hot_questions',0,top-1,withscores=True)
        return {'hot':[{'questions':q,'count':int(c)}for q,c in hot]}
    except Exception:
        return {'hot':[]}

@app.get('/pdfs')
async def list_pdfs():
    return {"pdfs":[{'name':f.name,'size':f.stat().st_size} for f in sorted(PDF_DIR.glob('*.pdf'))]}

@app.post('/upload_pdf')
async def upload_pdf(file:UploadFile=File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400,'只支持PDF')
    path = PDF_DIR/file.filename
    if path.exists():
        return {'ok':False,'message':'文件已存在'}
    try:
        with open(path,'wb') as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(500,f'失败:{e}')
    asyncio.create_task(_safe_rebuild())
    return {'ok':True,'message':'保存成功，后台正在重建中'}

@app.get('/pdf_file/{filename}')
async def get_pdf_file(filename:str):
    path = PDF_DIR/filename
    if not path.exists() or path.suffix.lower() != '.pdf':
        raise HTTPException(404,'不存在')
    return FileResponse(path,media_type='application/pdf')

@app.get('/context_by_page')
async def context_by_page(page:int=Query(...,ge=1),thread_id:Optional[str]=None):
    """如果page和缓存里的page一样就直接返回缓存所在页的contexts"""
    if thread_id:
        for msg in reversed(await get_history_cache_aside(thread_id)):
            if msg.get('role') == 'assistant' and msg.get('contexts'):
                for c in msg['contexts']:
                    if int(c.get('page',1)) == page:
                        return {'page':page,'context':c}
    return {'page':page,'context':None}

@app.get('/health')
async def health():
    return {'ok':agent_instance is not None,'redis':USE_REDIS,'pdf_count':len(list(PDF_DIR.glob('*.pdf')))}





