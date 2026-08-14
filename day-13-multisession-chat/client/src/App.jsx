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
  const visibleMessages = messages.filter((m) => m.role !== "system");
  const lastAssistantIndex = visibleMessages.map((m) => m.role).lastIndexOf("assistant");

  // The key layout decision (ChatGPT/Claude.ai-style): an empty chat --
  // either no session selected yet, or a freshly-created session with no
  // messages sent -- shows the input CENTERED on the page, with no
  // scrollable message area and no bottom bar. The moment there's at
  // least one visible message, the layout switches to the normal
  // scrollable-history + fixed-bottom-input arrangement. This is a
  // one-way transition per session: once a chat has messages, it never
  // goes back to the centered empty state (switching to a genuinely
  // different, still-empty session does show it centered again, since
  // that's a fresh empty conversation in its own right).
  const isEmptyChat = visibleMessages.length === 0;

  return (
    <div className="flex h-screen bg-bg">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />

      <div className="flex flex-col flex-1 min-w-0">
        {!isEmptyChat && (
          <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-bg/95 backdrop-blur px-6 py-4">
            <div>
              <h1 className="font-display text-base text-ink truncate max-w-md">
                {activeSession?.title || "New chat"}
              </h1>
              <p className="font-mono text-[10px] text-muted uppercase tracking-widest mt-0.5">
                {visibleMessages.length} messages
              </p>
            </div>
          </header>
        )}

        {isEmptyChat ? (
          // ---- CENTERED EMPTY STATE: input appears in the middle ----
          <main className="flex-1 flex flex-col items-center justify-center px-5 -mt-10">
            <div className="w-full max-w-2xl">
              <h2 className="font-display text-3xl text-ink text-center mb-8">
                What can I help with?
              </h2>
              <MessageInput onSend={handleSend} disabled={isLoading} variant="centered" />
              <ErrorBanner error={error} onDismiss={() => setError(null)} onRetry={null} />
            </div>
          </main>
        ) : (
          // ---- NORMAL STATE: scrollable history + fixed bottom input ----
          <>
            <main className="flex-1 overflow-y-auto px-5 py-8">
              <div className="max-w-3xl mx-auto">
                {visibleMessages.map((msg, i) => {
                  const isEmptyPlaceholder =
                    isLoading && i === visibleMessages.length - 1 && msg.role === "assistant" && msg.content === "";
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

            <MessageInput onSend={handleSend} disabled={isLoading || isRegenerating} variant="bottom" />
          </>
        )}
      </div>
    </div>
  );
}
