import re
import streamlit as st


def set_page(page_num):
    """回调函数：点击按钮时触发，更新 session_state 并触发 rerun"""
    st.session_state.target_page = page_num
    # 改完直接 rerun 刷新页面
    st.rerun()


def display_answer(answer: str, msg_idx: int = 0):
    """把 [Page X] 替换成真正的按钮，点击直接触发右侧 PDF 跳转"""
    parts = re.split(r'(\[Page\s+\d+\])', answer)

    # 用一个容器把文字和按钮包起来，避免排版散架
    col_text = st.container()

    for part in parts:
        m = re.match(r'\[Page\s+(\d+)\]', part)
        if m:
            page = int(m.group(1))
            # 这里用 st.button 代替 st.html，这是 Streamlit 原生支持跳转的唯一靠谱方式
            st.button(
                f"📖 Page {page}",
                key=f"btn_page_{page}_{msg_idx}_{id(part)}",
                on_click=set_page,
                args=(page,),
                use_container_width=False
            )
        else:
            if part.strip():
                with col_text:
                    st.markdown(part)


def handle_query_params():
    """清空 URL 参数，因为我们已经不走 URL 传参了"""
    pass
#整个流程：用户提问：“多头注意力是什么？”（见 app.py）1.大模型回答：“多头注意力是…… [Page 12]” 2.前端渲染：display_answer 把 [Page 12] 变成了一个 URL 为 ?target_page=12 的按钮。3.用户点击按钮：浏览器 URL 变成 localhost:8501/?target_page=12，触发页面重新加载。4.页面加载：app.py 第一步调用 handle_query_params()，把 12 存进 st.session_state.target_page。5.PDF 滚动：app.py 右侧的 pdf_viewer 读取 st.session_state.target_page，发现是 12，于是调用 scroll_to_page=12，PDF 自动翻到第 12 页。