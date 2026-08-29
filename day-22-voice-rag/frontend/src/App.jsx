import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

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

  // --- Voice / STT State ---
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

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
  }, [messages, isLoading, isTranscribing]);

  // Handle Recording Timer
  useEffect(() => {
    if (isRecording) {
      setRecordingSeconds(0);
      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  // Core Chat Execution
  const executeChatQuery = async (queryText) => {
    if (!queryText.trim() || isLoading) return;

    setMessages((prev) => [...prev, { role: "user", content: queryText }]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/rag/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: queryText,
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

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;
    const text = inputMessage.trim();
    setInputMessage("");
    executeChatQuery(text);
  };

  // --- Browser MediaRecorder Audio Capture ---
  const startRecording = async () => {
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        stream.getTracks().forEach((track) => track.stop());
        await processAudioForTranscription(audioBlob);
      };

      mediaRecorder.start(250); 
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone Access Error:", err);
      alert("Could not access microphone. Please ensure microphone permissions are granted.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processAudioForTranscription = async (audioBlob) => {
    if (audioBlob.size === 0) return;

    setIsTranscribing(true);
    const formData = new FormData();
    formData.append("file", audioBlob, "voice_query.webm");

    try {
      const res = await fetch(`${API_BASE}/api/transcribe`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        throw new Error(`Transcription failed with code ${res.status}`);
      }

      const data = await res.json();
      const transcribedText = data.text?.trim() || "";

      if (transcribedText) {
        // 1. Fill search bar with transcribed speech for user review
        setInputMessage((prev) => (prev ? prev + " " + transcribedText : transcribedText));
        
        // Note: Automatic executeChatQuery() removed. User must click Send.
      } else {
        alert("Whisper could not detect any speech in the audio recording. Please try again.");
      }
    } catch (err) {
      console.error("STT Error:", err);
      alert(`Voice Transcription Error: ${err.message}`);
    } finally {
      setIsTranscribing(false);
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
      <header className="rag-header">
        <div className="logo-title">
          <div className="status-dot"></div>
          <h1>Enterprise Voice RAG Control Plane</h1>
        </div>
        
        <div className="strategy-selector" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <label htmlFor="strategy-select" style={{ fontSize: "13px" }}>Strategy:</label>
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
              background: "#334155", 
              color: "#e2e8f0", 
              border: "none", 
              padding: "6px 12px", 
              borderRadius: "4px", 
              cursor: "pointer"
            }}
          >
            Reset
          </button>
        </div>
      </header>

      <main className="rag-main-layout">
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

        <section className="center-chat">
          <div className="chat-container">
            <div className="messages-scroll">
              {messages.length === 0 ? (
                <div className="placeholder-hero">
                  <h2>Enterprise Voice & Text RAG Assistant</h2>
                  <p>Ask technical questions via typing or click the mic button to speak.</p>
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
                    ⚡ Retrieving context and synthesizing answer...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Unified Input Bar: Text + Mic + Send */}
            <div className="unified-input-area" style={{ 
              display: "flex", flexDirection: "column", gap: "8px", 
              padding: "16px", background: "#1e293b", 
              borderTop: "1px solid #334155", borderRadius: "0 0 8px 8px" 
            }}>
              
              {/* Voice Status Indicator */}
              {(isRecording || isTranscribing) && (
                <div style={{ fontSize: "13px", fontWeight: "600", color: isRecording ? "#ef4444" : "#38bdf8", marginLeft: "4px" }}>
                  {isRecording ? `● Recording audio... ${recordingSeconds}s` : "🎙️ Processing audio through Whisper..."}
                </div>
              )}

              <form onSubmit={handleTextSubmit} style={{ display: "flex", gap: "10px", margin: 0 }}>
                
                {/* Text Input */}
                <input
                  type="text"
                  placeholder="Type a question or use the mic..."
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  disabled={isLoading || isTranscribing || isRecording}
                  style={{ 
                    flex: 1, padding: "12px 16px", borderRadius: "8px", 
                    border: "1px solid #334155", background: "#0f172a", 
                    color: "#f8fafc", fontSize: "15px" 
                  }}
                />

                {/* Mic Button */}
                <button
                  type="button"
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isLoading || isTranscribing}
                  title={isRecording ? "Stop Recording" : "Use Microphone"}
                  style={{
                    display: "flex", alignItems: "center", justifyContent: "center",
                    width: "45px", borderRadius: "8px", border: "none",
                    background: isRecording ? "#ef4444" : "#334155",
                    cursor: isLoading || isTranscribing ? "not-allowed" : "pointer",
                    boxShadow: isRecording ? "0 0 12px rgba(239, 68, 68, 0.5)" : "none",
                    transition: "all 0.2s ease"
                  }}
                >
                  <span style={{ fontSize: "18px" }}>{isRecording ? "⏹️" : "🎙️"}</span>
                </button>

                {/* Send Button */}
                <button 
                  type="submit" 
                  disabled={isLoading || isTranscribing || isRecording || !inputMessage.trim()}
                  style={{ 
                    padding: "0 20px", borderRadius: "8px", background: "#38bdf8", 
                    color: "#0f172a", fontWeight: "700", border: "none",
                    cursor: (isLoading || isTranscribing || isRecording || !inputMessage.trim()) ? "not-allowed" : "pointer",
                    opacity: (isLoading || isTranscribing || isRecording || !inputMessage.trim()) ? 0.6 : 1
                  }}
                >
                  Send
                </button>
              </form>
            </div>
          </div>
        </section>

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