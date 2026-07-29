"""FastAPI RAG Chatbot Server with Real-time Markdown & Token Streaming (ChatGPT Style)."""

import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# Import Pydantic models & RAG logic
from src.schemas import QueryRequest, QueryResponse
from src.rag import query_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI")

# Initialize FastAPI App
app = FastAPI(
    title="AMLA 2010 | RAG AI Assistant",
    description="Retrieval-Augmented Generation assistant for Anti-Money Laundering Act (AMLA 2010).",
    version="1.0.0"
)


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> str:
    """Serves the complete dark-mode chat interface with ChatGPT-style Markdown & Token Streaming."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lexicon | AMLA Knowledge Assistant</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/lucide@latest"></script>
        <!-- Added Marked.js for ChatGPT-style Markdown Rendering -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            :root {
                --bg-main: #121824;
                --bg-sidebar: #182030;
                --bg-card: #1f2a3e;
                --bg-input: #182030;
                --accent-teal: #2dd4bf;
                --accent-blue: #3b82f6;
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --border-color: #2a364f;
            }

            * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
            body { background-color: var(--bg-main); color: var(--text-primary); height: 100vh; display: flex; overflow: hidden; }

            /* Left Sidebar */
            .sidebar {
                width: 280px;
                background-color: var(--bg-sidebar);
                border-right: 1px solid var(--border-color);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                padding: 20px;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 12px;
                font-weight: 700;
                font-size: 1.15rem;
                color: #fff;
                margin-bottom: 24px;
            }

            .brand-icon {
                background: linear-gradient(135deg, #2563eb, #2dd4bf);
                padding: 8px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .btn-new-chat {
                width: 100%;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
                padding: 12px 16px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .btn-new-chat:hover { background: #27354d; border-color: #3d4f72; }

            .nav-section-title {
                font-size: 0.75rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin: 24px 0 12px 0;
            }

            .recent-list { list-style: none; display: flex; flex-direction: column; gap: 6px; }

            .recent-item {
                padding: 10px 12px;
                border-radius: 8px;
                font-size: 0.875rem;
                color: var(--text-secondary);
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: all 0.2s;
            }

            .recent-item:hover { background: var(--bg-card); color: var(--text-primary); }

            .sidebar-footer {
                border-top: 1px solid var(--border-color);
                padding-top: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .footer-link {
                color: var(--text-secondary);
                font-size: 0.875rem;
                text-decoration: none;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .footer-link:hover { color: var(--text-primary); }

            /* Main Workspace */
            .main-chat { flex: 1; display: flex; flex-direction: column; background: var(--bg-main); }

            .chat-header {
                padding: 18px 28px;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: rgba(24, 32, 48, 0.8);
            }

            .header-title { display: flex; align-items: center; gap: 12px; font-size: 1rem; font-weight: 600; }

            .status-badge {
                display: flex; align-items: center; gap: 6px; font-size: 0.8rem;
                color: var(--accent-teal); background: rgba(45, 212, 191, 0.12);
                padding: 4px 12px; border-radius: 20px; font-weight: 500;
            }

            .status-dot { width: 7px; height: 7px; background-color: var(--accent-teal); border-radius: 50%; }

            /* Chat Messages */
            .chat-messages {
                flex: 1; padding: 28px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px;
            }

            .message { display: flex; gap: 14px; max-width: 82%; }
            .message.user { align-self: flex-end; flex-direction: row-reverse; }
            .message.assistant { align-self: flex-start; }

            .avatar {
                width: 36px; height: 36px; border-radius: 10px;
                display: flex; align-items: center; justify-content: center; flex-shrink: 0;
            }

            .message.assistant .avatar {
                background: linear-gradient(135deg, #1e293b, #334155);
                border: 1px solid var(--border-color); color: var(--accent-teal);
            }

            .message.user .avatar { background: var(--accent-blue); color: #fff; }

            /* Bubble Styling & Markdown Elements */
            .bubble { padding: 14px 18px; border-radius: 14px; font-size: 0.935rem; line-height: 1.6; }
            .message.assistant .bubble { background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-primary); }
            .message.user .bubble { background: #1d4ed8; color: #ffffff; }

            /* Markdown Formatting Rules */
            .bubble h1, .bubble h2, .bubble h3 { color: var(--accent-teal); margin-top: 12px; margin-bottom: 6px; font-weight: 600; }
            .bubble h1 { font-size: 1.25rem; }
            .bubble h2 { font-size: 1.1rem; }
            .bubble h3 { font-size: 1.0rem; }
            .bubble p { margin-bottom: 8px; }
            .bubble p:last-child { margin-bottom: 0; }
            .bubble ul, .bubble ol { margin-left: 20px; margin-bottom: 8px; }
            .bubble li { margin-bottom: 4px; }
            .bubble strong { color: #ffffff; font-weight: 600; }

            /* Typing Cursor Indicator */
            .typing-cursor::after {
                content: '▋';
                display: inline-block;
                margin-left: 2px;
                color: var(--accent-teal);
                animation: blink 0.8s infinite;
            }

            @keyframes blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0; }
            }

            /* Blinking Dots Thinking Indicator */
            .typing-dots {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 4px 0;
            }

            .typing-dots span {
                width: 6px;
                height: 6px;
                background-color: var(--accent-teal);
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out both;
            }

            .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
            .typing-dots span:nth-child(2) { animation-delay: -0.16s; }

            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1.0); }
            }

            .sources-container { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-color); }
            .sources-title { font-size: 0.75rem; color: var(--text-secondary); font-weight: 600; margin-bottom: 8px; text-transform: uppercase; }
            .source-chips { display: flex; flex-wrap: wrap; gap: 8px; }
            .source-chip { background: #121824; border: 1px solid #2e3c54; color: var(--accent-teal); font-size: 0.78rem; padding: 4px 10px; border-radius: 6px; }

            /* Input Dock */
            .chat-input-area { padding: 20px 28px; background: var(--bg-main); border-top: 1px solid var(--border-color); }

            .input-wrapper {
                background: var(--bg-input); border: 1px solid var(--border-color);
                border-radius: 14px; padding: 8px 16px; display: flex; align-items: center; gap: 12px;
            }

            .input-wrapper:focus-within { border-color: var(--accent-blue); }

            .input-wrapper input {
                flex: 1; background: transparent; border: none; outline: none;
                color: var(--text-primary); font-size: 0.95rem; padding: 8px 0;
            }

            .btn-send {
                background: var(--accent-teal); color: #0f141c; border: none;
                padding: 10px 14px; border-radius: 10px; cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                font-weight: 600; transition: opacity 0.2s;
            }

            .btn-send:hover { opacity: 0.9; }
            .btn-send:disabled { opacity: 0.5; cursor: not-allowed; }
        </style>
    </head>
    <body>

        <aside class="sidebar">
            <div>
                <div class="brand">
                    <div class="brand-icon"><i data-lucide="book-open"></i></div>
                    <span>Lexicon | AMLA</span>
                </div>

                <button class="btn-new-chat" id="btnNewChat">
                    <i data-lucide="plus"></i>
                    <span>New Chat</span>
                </button>

                <div class="nav-section-title">Recent Chats</div>
                <ul class="recent-list">
                    <li class="recent-item" data-query="What constitutes money laundering under AMLA?">
                        <i data-lucide="message-square"></i>
                        <span>Money Laundering Definition</span>
                    </li>
                    <li class="recent-item" data-query="What are the reporting obligations for financial institutions?">
                        <i data-lucide="message-square"></i>
                        <span>Reporting Obligations</span>
                    </li>
                    <li class="recent-item" data-query="What are the penalties under Section 4 of AMLA?">
                        <i data-lucide="message-square"></i>
                        <span>Penalties & Punishments</span>
                    </li>
                </ul>
            </div>

            <div class="sidebar-footer">
                <a href="/docs" target="_blank" class="footer-link">
                    <i data-lucide="code"></i>
                    <span>FastAPI Docs</span>
                </a>
            </div>
        </aside>

        <main class="main-chat">
            <header class="chat-header">
                <div class="header-title">
                    <i data-lucide="book-marked" style="color: var(--accent-teal);"></i>
                    <span>Lexicon | AMLA Knowledge Assistant</span>
                </div>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    <span>Online</span>
                </div>
            </header>

            <div class="chat-messages" id="chatContainer">
                <div class="message assistant">
                    <div class="avatar"><i data-lucide="bot"></i></div>
                    <div class="bubble">Hello! I'm <strong>Lexicon</strong>. Ask me anything about Pakistan's <strong>Anti-Money Laundering Act (AMLA 2010)</strong>.</div>
                </div>
            </div>

            <div class="chat-input-area">
                <div class="input-wrapper">
                    <i data-lucide="paperclip" style="color: var(--text-secondary);"></i>
                    <input type="text" id="userInput" placeholder="Ask Lexicon a question about AMLA..." autocomplete="off">
                    <button class="btn-send" id="btnSend">
                        <i data-lucide="send"></i>
                    </button>
                </div>
            </div>
        </main>

        <script>
            document.addEventListener("DOMContentLoaded", () => {
                lucide.createIcons();

                const userInput = document.getElementById("userInput");
                const btnSend = document.getElementById("btnSend");
                const btnNewChat = document.getElementById("btnNewChat");
                const chatContainer = document.getElementById("chatContainer");

                btnSend.addEventListener("click", handleSend);
                userInput.addEventListener("keydown", (e) => {
                    if (e.key === "Enter") handleSend();
                });

                btnNewChat.addEventListener("click", () => {
                    chatContainer.innerHTML = `
                        <div class="message assistant">
                            <div class="avatar"><i data-lucide="bot"></i></div>
                            <div class="bubble">Hello! I'm <strong>Lexicon</strong>. Ask me anything about AMLA 2010.</div>
                        </div>
                    `;
                    lucide.createIcons();
                });

                document.querySelectorAll(".recent-item").forEach(item => {
                    item.addEventListener("click", () => {
                        const q = item.getAttribute("data-query");
                        if (q) {
                            userInput.value = q;
                            handleSend();
                        }
                    });
                });

                async function handleSend() {
                    const text = userInput.value.trim();
                    if (!text) return;

                    userInput.value = "";
                    btnSend.disabled = true;

                    // 1. Append User Message
                    appendUserMessage(text);

                    // 2. Append Assistant Placeholder with Blinking Thinking Dots
                    const { msgDiv, bubbleDiv } = createAssistantContainer();
                    bubbleDiv.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
                    chatContainer.scrollTop = chatContainer.scrollHeight;

                    try {
                        // 3. Perform Full RAG Query
                        const res = await fetch("/query", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ question: text })
                        });

                        if (!res.ok) {
                            const errData = await res.json().catch(() => ({}));
                            throw new Error(errData.detail || `Server error: ${res.status}`);
                        }

                        const data = await res.json();
                        const fullAnswer = data.answer || "No response generated.";

                        // 4. Word-by-Word Smooth Markdown Typing Animation (ChatGPT Style)
                        bubbleDiv.innerHTML = "";
                        bubbleDiv.classList.add("typing-cursor");

                        const words = fullAnswer.split(" ");
                        let currentText = "";

                        for (let i = 0; i < words.length; i++) {
                            currentText += (i === 0 ? "" : " ") + words[i];
                            // Parse Markdown via Marked library to convert **, #, and lists into HTML
                            bubbleDiv.innerHTML = marked.parse(currentText);
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                            // Pause 20ms between words to mimic LLM streaming speed
                            await new Promise(resolve => setTimeout(resolve, 20));
                        }

                        bubbleDiv.classList.remove("typing-cursor");

                        // 5. Append Citation Chips
                        if (data.sources && Array.isArray(data.sources) && data.sources.length > 0) {
                            const chips = data.sources.map(s => {
                                const page = (typeof s === "object" && s.page !== undefined) ? s.page : s;
                                return `<span class="source-chip">[Page ${page}]</span>`;
                            }).join(" ");

                            const sourcesDiv = document.createElement("div");
                            sourcesDiv.className = "sources-container";
                            sourcesDiv.innerHTML = `
                                <div class="sources-title">Sources Referenced</div>
                                <div class="source-chips">${chips}</div>
                            `;
                            bubbleDiv.appendChild(sourcesDiv);
                        }

                    } catch (err) {
                        bubbleDiv.innerHTML = `<span style="color: #f87171;">❌ Error: ${escapeHtml(err.message)}</span>`;
                    } finally {
                        btnSend.disabled = false;
                        userInput.focus();
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }

                function appendUserMessage(text) {
                    const msgDiv = document.createElement("div");
                    msgDiv.className = "message user";
                    msgDiv.innerHTML = `<div class="avatar"><i data-lucide="user"></i></div><div class="bubble">${escapeHtml(text)}</div>`;
                    chatContainer.appendChild(msgDiv);
                    lucide.createIcons();
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                function createAssistantContainer() {
                    const msgDiv = document.createElement("div");
                    msgDiv.className = "message assistant";
                    const avatarDiv = document.createElement("div");
                    avatarDiv.className = "avatar";
                    avatarDiv.innerHTML = `<i data-lucide="bot"></i>`;

                    const bubbleDiv = document.createElement("div");
                    bubbleDiv.className = "bubble";

                    msgDiv.appendChild(avatarDiv);
                    msgDiv.appendChild(bubbleDiv);
                    chatContainer.appendChild(msgDiv);
                    lucide.createIcons();
                    return { msgDiv, bubbleDiv };
                }

                function escapeHtml(str) {
                    if (!str) return "";
                    return String(str)
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;");
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health_check() -> dict[str, str]:
    """Verify service health status."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def ask_question(request: QueryRequest) -> QueryResponse:
    """Query endpoint accepting and returning validated Pydantic models."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        # Pass request into query_rag execution handler
        response_obj = query_rag(request)
        
        # If query_rag returns a dict or plain text, convert to QueryResponse
        if isinstance(response_obj, dict):
            return QueryResponse(**response_obj)
        elif isinstance(response_obj, str):
            return QueryResponse(
                question=request.question,
                answer=response_obj,
                sources=[],
                is_in_scope=True
            )
        return response_obj
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)