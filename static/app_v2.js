const { API_PREFIX, DEFAULT_PDF } = window.__APP__;

const $ = (id) => document.getElementById(id);
const state = {
  threadId: localStorage.getItem("thread_id") || null,
  currentPage: 1,
  pdfName: DEFAULT_PDF || "",
  streaming: false,
};

function pdfUrl(name, page) {
  return `${API_PREFIX}/pdf_file/${encodeURIComponent(name)}#page=${page}&zoom=page-width`;
}

function loadPdf(name, page = 1) {
  state.pdfName = name;
  state.currentPage = page;
  $("pdfName").textContent = name || "PDF 预览";
  $("pdfFrame").src = name ? pdfUrl(name, page) : "";
  updatePageIndicator();
}

function jumpToPage(page) {
  if (!state.pdfName) return;
  state.currentPage = page;

  const oldIframe = $("pdfFrame");
  const parent = oldIframe.parentNode;
  const newUrl = pdfUrl(state.pdfName, page);

  const newIframe = document.createElement("iframe");
  newIframe.id = "pdfFrame";
  newIframe.className = oldIframe.className;
  newIframe.style.width = "100%";
  newIframe.style.height = "100%";
  newIframe.style.border = "none";
  newIframe.src = newUrl;

  parent.replaceChild(newIframe, oldIframe);

  updatePageIndicator();

  document.querySelectorAll(".page-btn").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.page) === page);
  });
}

function updatePageIndicator() {
  $("pageIndicator").textContent = `Page ${state.currentPage}`;
}

$("prevPage").onclick = () => jumpToPage(Math.max(1, state.currentPage - 1));
$("nextPage").onclick = () => jumpToPage(state.currentPage + 1);

function renderAnswerInto(container, answer) {
  container.innerHTML = "";
  const parts = answer.split(/(\[Page\s+\d+\])/g);
  for (const part of parts) {
    const m = part.match(/^\[Page\s+(\d+)\]$/);
    if (m) {
      const page = parseInt(m[1], 10);
      const btn = document.createElement("button");
      btn.className = "page-btn";
      btn.dataset.page = page;
      btn.textContent = `📖 Page ${page}`;
      btn.onclick = () => jumpToPage(page);
      container.appendChild(btn);
    } else if (part.trim()) {
      const div = document.createElement("div");
      // 如果 marked 没加载成功，就直接显示纯文本，防止网页死机
      div.innerHTML = typeof marked !== 'undefined' ? marked.parse(part) : part;
      container.appendChild(div);
    }
  }
}

function renderContexts(parent, contexts) {
  if (!contexts || !contexts.length) return;
  // 防止重复添加参考资料面板
  if (parent.querySelector('.contexts')) return;

  const wrap = document.createElement("div");
  wrap.className = "contexts";
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `参考资料（${contexts.length}）`;
  details.appendChild(summary);

  contexts.forEach((c) => {
    const item = document.createElement("div");
    item.className = "ctx-item";
    const meta = document.createElement("div");
    meta.className = "meta";
    const score = typeof c.score === "number" ? c.score.toFixed(3) : "-";
    meta.textContent = `${c.source || "unknown"} · Page ${c.page} · score=${score}`;
    const text = document.createElement("div");
    text.className = "text";
    text.textContent = (c.text || "").slice(0, 200) + "...";
    item.appendChild(meta);
    item.appendChild(text);
    item.style.cursor = "pointer";
    item.onclick = () => jumpToPage(Number(c.page) || 1);
    details.appendChild(item);
  });

  wrap.appendChild(details);
  parent.appendChild(wrap);
}

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  div.appendChild(bubble);
  $("history").appendChild(div);
  scrollHistoryToBottom();
  return bubble;
}

function appendAssistantMessage() {
  const div = document.createElement("div");
  div.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  div.appendChild(bubble);
  $("history").appendChild(div);
  scrollHistoryToBottom();
  return bubble;
}

function scrollHistoryToBottom() {
  const h = $("history");
  h.scrollTop = h.scrollHeight;
}

function renderHistory(history) {
  $("history").innerHTML = "";
  history.forEach((msg) => {
    const div = document.createElement("div");
    div.className = `msg ${msg.role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    div.appendChild(bubble);
    if (msg.role === "assistant") {
      renderAnswerInto(bubble, msg.content || "");
      renderContexts(div, msg.contexts);
    } else {
      bubble.textContent = msg.content || "";
    }
    $("history").appendChild(div);
  });
  scrollHistoryToBottom();
}

async function loadHistory() {
  if (!state.threadId) return;
  try {
    const resp = await fetch(`${API_PREFIX}/history/${state.threadId}`);
    const data = await resp.json();
    renderHistory(data.history || []);
  } catch (e) {
    console.warn("load history failed:", e);
  }
}

// ============ 修复了这里的函数闭合问题 ============
async function ask(question) {
  if (state.streaming) return;
  state.streaming = true;
  $("sendBtn").disabled = true;
  $("status").textContent = "正在生成…";

  appendUserMessage(question);
  const bubble = appendAssistantMessage();

  let fullAnswer = "";
  let contexts = [];

  const body = JSON.stringify({
    question,
    thread_id: state.threadId,
    use_cache: true,
  });

  try {
    const resp = await fetch(`${API_PREFIX}/chat_stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const blocks = buf.split("\n\n");
      buf = blocks.pop();

      for (const block of blocks) {
        let event = "message", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }

        const payload = data ? JSON.parse(data) : {};

        if (event === "meta") {
          state.threadId = payload.thread_id;
          localStorage.setItem("thread_id", state.threadId);
        } else if (event === "token") {
          fullAnswer += payload.content || "";
          bubble.innerHTML = typeof marked !== 'undefined' ? marked.parse(fullAnswer) : fullAnswer;
          scrollHistoryToBottom();
        } else if (event === "contexts") {
          contexts = payload.contexts || [];
        } else if (event === "done") {
          $("status").textContent = payload.cached ? "⚡ 命中缓存" : "";
          renderAnswerInto(bubble, fullAnswer);
        } else if (event === "error") {
          $("status").textContent = `❌ ${payload.message}`;
          bubble.textContent = `出错了：${payload.message}`;
        }
      }
    }

    renderContexts(bubble.parentElement, contexts);
    loadHotQuestions();
  } catch (e) {
    $("status").textContent = `❌ ${e.message}`;
    bubble.textContent = `请求失败：${e.message}`;
  } finally {
    state.streaming = false;
    $("sendBtn").disabled = false;
  }
}
// =================================================

async function loadHotQuestions() {
  try {
    const resp = await fetch(`${API_PREFIX}/hot_questions?top=5`);
    const data = await resp.json();
    const ol = $("hotList");
    ol.innerHTML = "";
    if (!data.hot || !data.hot.length) {
      ol.innerHTML = "<li style='color:#9ca3af'>暂无提问记录</li>";
      return;
    }
    data.hot.forEach(({ question, count }) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = "#";
      a.textContent = `${question.length > 24 ? question.slice(0, 24) + "..." : question} (${count})`;
      a.onclick = (ev) => {
        ev.preventDefault();
        $("question").value = question;
        $("question").focus();
      };
      li.appendChild(a);
      ol.appendChild(li);
    });
  } catch (e) {
    console.warn("hot questions failed:", e);
  }
}

async function uploadPdf(file) {
  const fd = new FormData();
  fd.append("file", file);
  $("status").textContent = "正在上传…";
  try {
    const resp = await fetch(`${API_PREFIX}/upload_pdf`, { method: "POST", body: fd });
    const data = await resp.json();
    $("status").textContent = data.message || (data.ok ? "上传成功" : "上传失败");
    if (data.ok) {
      setTimeout(() => loadPdf(file.name, 1), 500);
    }
  } catch (e) {
    $("status").textContent = `上传失败：${e.message}`;
  }
}

$("sendBtn").onclick = () => {
  const q = $("question").value.trim();
  if (!q) return;
  $("question").value = "";
  ask(q);
};

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("sendBtn").click();
  }
});

$("newThreadBtn").onclick = () => {
  state.threadId = null;
  localStorage.removeItem("thread_id");
  $("history").innerHTML = "";
  $("status").textContent = "已新建会话";
};

$("pdfUpload").onchange = (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) uploadPdf(f);
  e.target.value = "";
};

(async function init() {
  loadPdf(state.pdfName, 1);
  await loadHistory();
  loadHotQuestions();
})();