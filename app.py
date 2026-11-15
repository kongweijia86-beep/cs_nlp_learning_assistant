import uuid
import hashlib
import asyncio
import json
import redis
from pathlib import Path
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_core.messages import HumanMessage
from src.rag import RAG
from src.agent_graph import build_agent
from src.citation_ui import display_answer, handle_query_params

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / 'data' / 'raw'
DEFAULT_PDF = PDF_DIR / "Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf"

USE_REDIS = True
try:
    if USE_REDIS:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, protocol=2)
        r.ping()
except Exception:
    USE_REDIS = False
    st.toast("⚠️ 未连接到 Redis，缓存及持久化功能已禁用", icon="⚠️")


def render_contexts(contexts):
    """渲染参考资料"""
    if not contexts:
        return []
    with st.expander('参考资料', expanded=True):
        for item in contexts:
            st.write(f"**{item['source']}-Page{item['page']}** (score={item.get('score', 0.0):.3f})")
            st.caption(item['text'][:200] + '...')

def save_message_to_redis(thread_id, role, content, extra_data=None):
    """保存单条消息到 Redis List"""
    if not USE_REDIS:
        return
    key = f"chat_memory:{thread_id}"
    message = {'role': role, 'content': content}
    if extra_data:
        message.update(extra_data)

    try:
        r.rpush(key, json.dumps(message, ensure_ascii=False))
        r.ltrim(key, -20, -1)
        r.expire(key, 7 * 24 * 3600)
    except Exception:
        pass


def load_history_from_redis(thread_id):
    """从 Redis List 加载全部聊天记录"""
    if not USE_REDIS:
        return []
    key = f"chat_memory:{thread_id}"
    try:
        raw_msgs = r.lrange(key, 0, -1)
        return [json.loads(m) for m in raw_msgs]
    except Exception:
        return []


def main():
    st.set_page_config(page_title='NLP学习助手', page_icon='🧠', layout="wide")
    handle_query_params()

    st.title('🧠大模型学习助手')
    st.caption('基于 Tool Calling 的 Agent (支持动态 PDF 上传、流式输出与缓存)')

    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_history_from_redis(st.session_state.thread_id)

    with st.sidebar:
        st.header("知识库管理")
        uploaded_file = st.file_uploader('上传新的 PDF 文档', type=['pdf'])
        if uploaded_file is not None:
            save_path = PDF_DIR / uploaded_file.name
            if not save_path.exists():
                with open(save_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                    st.success(f"文件 {uploaded_file.name} 已经保存")
                    st.cache_resource.clear()
                    st.rerun()

        st.divider()

        if USE_REDIS:
            st.header("🔥 热门提问 Top 5")
            try:
                hot_list = r.zrevrange("hot_questions", 0, 4, withscores=True)
                if hot_list:
                    for idx, (q, count) in enumerate(hot_list):
                        short_q = q[:20] + "..." if len(q) > 20 else q
                        st.write(f"{idx + 1}. {short_q} ({int(count)} 次)")
                else:
                    st.caption("暂无提问记录")
            except Exception:
                pass
            st.divider()

        st.header('设置')
        if st.button("新建会话(清空回忆)"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()

    @st.cache_resource
    def load_resources(pdf_dir_str, current_files_str):
        rag = RAG(pdf_dir_str)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent = loop.run_until_complete(build_agent(rag))
        return rag, agent

    with st.spinner("正在加载知识库并编译 Agent 图..."):
        current_files = sorted([f.name for f in PDF_DIR.glob('*pdf')])
        try:
            rag, agent = load_resources(str(PDF_DIR), str(current_files))
        except Exception as e:
            st.error(f"知识库加载失败: {e}")
            st.stop()

    col_chat, col_pdf = st.columns([1, 1])

    with col_chat:
        # 渲染历史记录
        for idx, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg['role']):
                if msg['role'] == 'assistant':
                    display_answer(msg['content'], msg_idx=idx)
                else:
                    st.write(msg['content'])
                if msg['role'] == 'assistant' and 'contexts' in msg:
                    render_contexts(msg['contexts'])

        question = st.chat_input('请输入你的问题')
        if question:
            if USE_REDIS:
                try:
                    r.zincrby("hot_questions", 1, question.strip())
                except Exception:
                    pass

            q_key = ""
            if USE_REDIS:
                q_key = f"rag_cache:{hashlib.md5(question.strip().lower().encode('utf-8')).hexdigest()}"
                try:
                    cached_answer = r.get(q_key)

                    if cached_answer == "__NULL__":
                        st.warning("这个问题之前查过，资料库里没有（请等待 1 分钟后再试）")
                        user_msg = {'role': 'user', 'content': question}
                        st.session_state.chat_history.append(user_msg)
                        with st.chat_message('user'):
                            st.write(question)
                        with st.chat_message('assistant'):
                            st.info("资料不足，无法回答。")
                        st.stop()  #防止缓存穿透

                    elif cached_answer:
                        st.success("命中本地 Redis 缓存")

                        user_msg = {'role': 'user', 'content': question}
                        st.session_state.chat_history.append(user_msg)
                        save_message_to_redis(st.session_state.thread_id, 'user', question)

                        with st.chat_message('user'):
                            st.write(question)

                        with st.chat_message('assistant'):
                            display_answer(cached_answer, msg_idx=len(st.session_state.chat_history))

                        assistant_msg = {'role': 'assistant', 'content': cached_answer, 'contexts': []}
                        st.session_state.chat_history.append(assistant_msg)
                        save_message_to_redis(st.session_state.thread_id, 'assistant', cached_answer)

                        st.rerun()
                except Exception:
                    pass

            # 渲染用户消息
            user_msg = {'role': 'user', 'content': question}
            st.session_state.chat_history.append(user_msg)
            save_message_to_redis(st.session_state.thread_id, 'user', question)

            with st.chat_message('user'):
                st.write(question)

            # 流式输出区
            with st.chat_message('assistant'):
                final_state_data = {}

                def sync_stream_wrapper():
                    config = {'configurable': {'thread_id': st.session_state.thread_id}}
                    inputs = {
                        'messages': [HumanMessage(content=question)],
                        'thread_id': st.session_state.thread_id  # 用于长期记忆
                    }

                    async def _async_stream():
                        async for event in agent.astream_events(inputs, config=config, version="v2"):
                            kind = event["event"]
                            if kind == "on_chat_model_stream":
                                content = event["data"]["chunk"].content
                                if content:
                                    yield content
                            elif kind == "on_chain_end":
                                output = event["data"].get("output", {})
                                if isinstance(output, dict) and "answer" in output:
                                    final_state_data.update(output)

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        async_gen = _async_stream()
                        while True:
                            try:
                                chunk = loop.run_until_complete(async_gen.__anext__())
                                yield chunk
                            except StopAsyncIteration:
                                break
                    finally:
                        loop.close()

                full_answer = st.write_stream(sync_stream_wrapper())

                if USE_REDIS and full_answer:
                    try:
                        if "资料中没有找到" in full_answer or "资料不足" in full_answer:
                            # 缓存空结果 60 秒，防止缓存穿透
                            r.set(q_key, "__NULL__", ex=60)
                        else:
                            # 正常结果缓存 1 小时
                            r.set(q_key, full_answer, ex=3600)
                    except Exception:
                        pass
                contexts = final_state_data.get("contexts", [])
                render_contexts(contexts)

            # 存入会话历史
            assistant_msg = {
                'role': 'assistant',
                'content': full_answer,
                'contexts': contexts
            }
            st.session_state.chat_history.append(assistant_msg)

            save_message_to_redis(
                st.session_state.thread_id,
                'assistant',
                full_answer,
                extra_data={'contexts': contexts}
            )

            st.rerun()

    with col_pdf:
        st.subheader('PDF预览')
        target_pdf = str(DEFAULT_PDF)
        if not DEFAULT_PDF.exists():
            pdf_files = list(PDF_DIR.glob("*.pdf"))
            if pdf_files:
                target_pdf = str(pdf_files[0])
            else:
                st.warning("没有找到任何 PDF 文件")
                return

        target_page = st.session_state.get('target_page', 1)
        pdf_viewer(
            input=target_pdf,
            width=600,
            height=900,
            scroll_to_page=target_page,
            key=f"pdf_viewer_page_{target_page}"
        )


if __name__ == '__main__':
    main()
