/**
 * App.jsx (Day 13: multi-session rewrite)
 * -------------------------------------------
 * Manages MULTIPLE independent chat sessions, each with its own message
 * history, AI-generated title, and isolated state -- rather than Day
 * 12's single active conversation.
 *
 * STATE SHAPE:
 *   sessions: [{ id, title, messages: [{role, content, timestamp}], createdAt, lastActiveAt }]
 *   activeSessionId: the currently-displayed session's id
 *
 * PERSISTENCE: every change to `sessions` is written to localStorage
 * (via sessionStorage.js) so a full page refresh restores every
 * conversation exactly as it was -- Day 13's requirement.
 */

import { useState, useRef, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ChatMessage from "./components/ChatMessage";
import MessageInput from "./components/MessageInput";
import TypingIndicator from "./components/TypingIndicator";
import ErrorBanner from "./components/ErrorBanner";
import { streamMessage, generateTitle, regenerateResponse, ChatApiError } from "./api/chatApi";
import { loadSessions, saveSessions } from "./lib/sessionStorage";

function createNewSession() {
  const id = crypto.randomUUID();
  const now = new Date().toISOString();
  return { id, title: null, messages: [], createdAt: now, lastActiveAt: now };
}

export default function App() {
  const [sessions, setSessions] = useState(() => loadSessions());
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [error, setError] = useState(null);

  const scrollAnchorRef = useRef(null);

  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    if (sessions.length > 0 && activeSessionId === null) {
      const mostRecent = [...sessions].sort(
        (a, b) => new Date(b.lastActiveAt) - new Date(a.lastActiveAt)
      )[0];
      setActiveSessionId(mostRecent.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeSession?.messages, isLoading]);

  function updateSession(sessionId, updater) {
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? updater(s) : s)));
  }

  function handleNewChat() {
    const newSession = createNewSession();
    setSessions((prev) => [...prev, newSession]);
    setActiveSessionId(newSession.id);
    setError(null);
  }

  function handleSelectSession(sessionId) {
    setActiveSessionId(sessionId);
    setError(null);
  }

  async function handleSend(text) {
    setError(null);

    let targetSessionId = activeSessionId;
    if (!targetSessionId) {
      const newSession = createNewSession();
      setSessions((prev) => [...prev, newSession]);
      setActiveSessionId(newSession.id);
      targetSessionId = newSession.id;
    }

    const userMessage = { role: "user", content: text, timestamp: new Date().toISOString() };
    const wasFirstMessage =
      (sessions.find((s) => s.id === targetSessionId)?.messages.length || 0) === 0;

    // The assistant's message starts EMPTY and streams in incrementally
    // (Day 12's improvement, carried forward) -- appended right alongside
    // the user's message so both show up in the log immediately.
    updateSession(targetSessionId, (s) => ({
      ...s,
      messages: [...s.messages, userMessage, { role: "assistant", content: "", timestamp: new Date().toISOString() }],
      lastActiveAt: new Date().toISOString(),
    }));
    setIsLoading(true);

    let receivedFirstChunk = false;

    function handleChunk(chunkText) {
      if (!receivedFirstChunk) {
        receivedFirstChunk = true;
        setIsLoading(false);
      }
      updateSession(targetSessionId, (s) => {
        const messages = [...s.messages];
        const lastIndex = messages.length - 1;
        messages[lastIndex] = { ...messages[lastIndex], content: messages[lastIndex].content + chunkText };
        return { ...s, messages };
      });
    }

    try {
      await streamMessage(text, targetSessionId, handleChunk);

      if (wasFirstMessage) {
        generateTitle(targetSessionId)
          .then(({ title }) => {
            updateSession(targetSessionId, (s) => ({ ...s, title }));
          })
          .catch((err) => {
            console.warn("Title generation failed:", err);
          });
      }
    } catch (err) {
      // Remove the empty/partial assistant placeholder on failure, rather
      // than leaving a blank or half-written message in the session.
      updateSession(targetSessionId, (s) => ({ ...s, messages: s.messages.slice(0, -1) }));

      if (err instanceof ChatApiError) {
        setError({ message: err.message, status: err.status });
      } else {
        setError({ message: "An unexpected error occurred.", status: null });
      }
    } finally {
      setIsLoading(false);
    }
  }

  async function handleRegenerate() {
    if (!activeSessionId) return;
    setError(null);
    setIsRegenerating(true);

    updateSession(activeSessionId, (s) => {
      const messages = [...s.messages];
      if (messages.length > 0 && messages[messages.length - 1].role === "assistant") {
        messages.pop();
      }
      return { ...s, messages };
    });

    try {
      const data = await regenerateResponse(activeSessionId);
      updateSession(activeSessionId, (s) => ({
        ...s,
        messages: [...s.messages, { role: "assistant", content: data.response, timestamp: new Date().toISOString() }],
        lastActiveAt: new Date().toISOString(),
      }));
    } catch (err) {
      if (err instanceof ChatApiError) {
        setError({ message: err.message, status: err.status });
      } else {
        setError({ message: "An unexpected error occurred.", status: null });
      }
    } finally {
      setIsRegenerating(false);
    }
  }

  const messages = activeSession?.messages || [];
  const lastAssistantIndex = [...messages].map((m) => m.role).lastIndexOf("assistant");

  return (
    <div className="flex h-screen bg-white">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />

      <div className="flex flex-col flex-1 min-w-0">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/95 backdrop-blur px-6 py-3.5">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
              C
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-800 truncate max-w-md">
                {activeSession?.title || (activeSession ? "New conversation" : "Chat Assistant")}
              </h1>
              <p className="text-xs text-slate-400">
                {activeSession
                  ? `${messages.filter((m) => m.role !== "system").length} messages`
                  : `${sessions.length} conversation${sessions.length === 1 ? "" : "s"} saved`}
              </p>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto">
            {!activeSession && (
              <div className="flex flex-col items-center justify-center text-center text-slate-400 mt-20">
                <p className="text-sm">Click "New Chat" or just start typing to begin.</p>
              </div>
            )}

            {activeSession && messages.length === 0 && !isLoading && (
              <div className="flex flex-col items-center justify-center text-center text-slate-400 mt-20">
                <p className="text-sm">Send a message to start this conversation.</p>
              </div>
            )}

            {messages.map((msg, i) => {
              const isEmptyPlaceholder =
                isLoading && i === messages.length - 1 && msg.role === "assistant" && msg.content === "";
              if (isEmptyPlaceholder) return null;

              return (
                <ChatMessage
                  key={i}
                  role={msg.role}
                  content={msg.content}
                  timestamp={msg.timestamp}
                  isLastAssistantMessage={msg.role === "assistant" && i === lastAssistantIndex}
                  onRegenerate={handleRegenerate}
                  isRegenerating={isRegenerating}
                />
              );
            })}

            {isLoading && <TypingIndicator />}

            <div ref={scrollAnchorRef} />
          </div>
        </main>

        <ErrorBanner error={error} onDismiss={() => setError(null)} onRetry={null} />

        <MessageInput onSend={handleSend} disabled={isLoading || isRegenerating} />
      </div>
    </div>
  );
}
