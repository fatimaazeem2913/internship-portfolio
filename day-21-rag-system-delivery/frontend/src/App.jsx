import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css"; // Imports the math styling

const API_BASE = "http://localhost:8000";

export default function App() {
  const [sessionId] = useState(() => "session_" + Math.random().toString(36).substring(2, 9));
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [strategy, setStrategy] = useState("hybrid");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [sources, setSources] = useState([]);
  const [activeCitations, setActiveCitations] = useState([]);
  const [activeChunks, setActiveChunks] = useState([]);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const fetchSources = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/rag/sources`);
      if (res.ok) {
        const data = await res.json();
        setSources(data.sources || []);
      }
    } catch (e) {
      console.error("Failed to load sources:", e);
    }
  };

  useEffect(() => {
    fetchSources();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userText = inputMessage.trim();
    setInputMessage("");
    
    // Add user message to state
    setMessages((prev) => [...prev, { role: "user", content: userText }]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/rag/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: userText,
          strategy: strategy
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          standalone_query: data.standalone_query,
          citations: data.citations || [],
          retrieved_chunks: data.retrieved_chunks || []
        }
      ]);

      setActiveCitations(data.citations || []);
      setActiveChunks(data.retrieved_chunks || []);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: Unable to complete request. ${err.message}`,
          citations: [],
          retrieved_chunks: []
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus(`Indexing ${file.name}...`);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/rag/ingest`, {
        method: "POST",
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        setUploadStatus(`✓ Indexed ${data.indexed_chunks} chunks from ${data.filename}`);
        fetchSources();
      } else {
        setUploadStatus("✗ Upload failed. Check file format.");
      }
    } catch (e) {
      setUploadStatus(`✗ Upload error: ${e.message}`);
    }
  };

  const handleResetSession = async () => {
    try {
      await fetch(`${API_BASE}/api/rag/session/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      setMessages([]);
      setActiveCitations([]);
      setActiveChunks([]);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="rag-app-container">
      {/* Header */}
      <header className="rag-header">
        <div className="logo-title">
          <div className="status-dot"></div>
          <h1>Enterprise RAG Control Plane</h1>
        </div>
        <div className="strategy-selector">
          <label htmlFor="strategy-select">Retrieval Strategy:</label>
          <select
            id="strategy-select"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
          >
            <option value="hybrid">Hybrid Search (RRF)</option>
            <option value="dense">Dense Vector (MiniLM)</option>
            <option value="bm25">Sparse Keyword (BM25)</option>
            <option value="hierarchical">Hierarchical Compressor</option>
          </select>
          <button 
            onClick={handleResetSession} 
            style={{
              marginLeft: "12px", 
              background: "#334155", 
              color: "#e2e8f0", 
              border: "none", 
              padding: "6px 12px", 
              borderRadius: "4px", 
              cursor: "pointer"
            }}
          >
            Reset Session
          </button>
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <main className="rag-main-layout">
        {/* Left Column: Ingestion & Indexed Corpus */}
        <aside className="left-panel">
          <div className="upload-card">
            <h3>Document Ingestion</h3>
            <div className="dropzone-box">
              <input
                type="file"
                id="doc-upload"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".pdf,.docx,.txt"
              />
              <label htmlFor="doc-upload" className="dropzone-label">
                + Upload PDF, DOCX, or TXT
              </label>
            </div>
            {uploadStatus && <p className="status-caption">{uploadStatus}</p>}
          </div>

          <div className="sources-list-card">
            <h3>Indexed Corpus ({sources.length})</h3>
            {sources.length === 0 ? (
              <p className="empty-sub">No external documents indexed.</p>
            ) : (
              <ul>
                {sources.map((src, idx) => (
                  <li key={idx} className="source-item">
                    <span className="source-name" title={src.filename}>{src.filename}</span>
                    <span className="source-badge">{src.total_chunks} chunks</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        {/* Center Column: Multi-Turn Conversation */}
        <section className="center-chat">
          <div className="chat-container">
            <div className="messages-scroll">
              {messages.length === 0 ? (
                <div className="placeholder-hero">
                  <h2>Enterprise RAG Q&A Assistant</h2>
                  <p>Ask technical questions, formulas, or policy inquiries. Verified citations will attach automatically.</p>
                </div>
              ) : (
                messages.map((m, idx) => (
                  <div key={idx} className={`message-row ${m.role}`}>
                    <div className="message-bubble">
                      {m.standalone_query && m.standalone_query !== m.content && (
                        <div className="reformulation-tag">
                          🔍 Reformulated: "{m.standalone_query}"
                        </div>
                      )}
                      {/* Markdown and LaTeX rendering block */}
                      <div style={{ whiteSpace: "pre-wrap", lineHeight: "1.6" }}>
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {m.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                ))
              )}
              {isLoading && (
                <div className="message-row assistant">
                  <div className="message-bubble" style={{ color: "#94a3b8" }}>
                    Retrieving context and synthesizing answer...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <form className="chat-input-bar" onSubmit={handleSendMessage}>
              <input
                type="text"
                placeholder="Ask about Neural Networks, MSE, SVM, SVD, or API limits..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                disabled={isLoading}
              />
              <button type="submit" disabled={isLoading || !inputMessage.trim()}>
                Send
              </button>
            </form>
          </div>
        </section>

        {/* Right Column: Citation & Chunk Inspector */}
        <aside className="right-sidebar">
          <div className="sidebar-container">
            <div className="sidebar-section">
              <h3>Active Citations</h3>
              {activeCitations.length === 0 ? (
                <p className="empty-msg">No active citations for current query.</p>
              ) : (
                <div className="citation-pill-box">
                  {activeCitations.map((cit, idx) => (
                    <span key={idx} className="citation-pill">{cit}</span>
                  ))}
                </div>
              )}
            </div>

            <div className="sidebar-section">
              <h3>Retrieved Chunks ({activeChunks.length})</h3>
              {activeChunks.length === 0 ? (
                <p className="empty-msg">Retrieved source passages will appear here.</p>
              ) : (
                <div className="chunks-list">
                  {activeChunks.map((chunk, idx) => (
                    <div key={idx} className="chunk-card">
                      <div className="chunk-header">
                        <span>Page {chunk.page}</span>
                        <span>{chunk.source}</span>
                      </div>
                      <div className="chunk-body">{chunk.content}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}