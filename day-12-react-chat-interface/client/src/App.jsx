/**
 * App.jsx
 * ----------
 * The top-level component wiring everything together:
 *   - Chat message log (useState)
 *   - Session ID persistence across turns (Day 11's session_id concept,
 *     now on the frontend -- the SAME id must be resent every request)
 *   - Loading state driving the typing indicator
 *   - Error state driving the error banner
 *
 * LAYOUT: a sticky header, a scrollable middle chat area (flex-1 +
 * overflow-y-auto), and a fixed input bar at the bottom -- the standard
 * three-region chat app layout, built with plain flexbox (no absolute
 * positioning hacks needed).
 */

import { useState, useRef, useEffect } from "react";
import ChatMessage from "./components/ChatMessage";
import MessageInput from "./components/MessageInput";
import TypingIndicator from "./components/TypingIndicator";
import ErrorBanner from "./components/ErrorBanner";
import { streamMessage, ChatApiError } from "./api/chatApi";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);

  const scrollAnchorRef = useRef(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  async function handleSend(text) {
    setError(null);

    const userMessage = { role: "user", content: text, timestamp: new Date().toISOString() };

    // The assistant's message starts EMPTY and is appended to the log
    // immediately -- its content gets filled in incrementally as chunks
    // arrive (below), rather than appearing all at once only after the
    // full reply finishes generating.
    setMessages((prev) => [...prev, userMessage, { role: "assistant", content: "", timestamp: new Date().toISOString() }]);
    setIsLoading(true);

    // Tracks whether the FIRST chunk has arrived yet, purely to know when
    // to hide the typing indicator -- once real text starts appearing in
    // the message bubble itself, the separate "typing..." dots are
    // redundant and should disappear.
    let receivedFirstChunk = false;

    function handleChunk(text) {
      if (!receivedFirstChunk) {
        receivedFirstChunk = true;
        setIsLoading(false);
      }
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        updated[lastIndex] = { ...updated[lastIndex], content: updated[lastIndex].content + text };
        return updated;
      });
    }

    try {
      const data = await streamMessage(text, sessionId, handleChunk);

      if (!sessionId) {
        setSessionId(data.session_id);
      }
      setLastFailedMessage(null);
    } catch (err) {
      // Remove the empty/partial assistant placeholder bubble on failure,
      // rather than leaving a blank or half-written message in the log.
      setMessages((prev) => prev.slice(0, -1));

      if (err instanceof ChatApiError) {
        setError({ message: err.message, status: err.status });
      } else {
        setError({ message: "An unexpected error occurred.", status: null });
      }
      setLastFailedMessage(text);
    } finally {
      setIsLoading(false);
    }
  }

  function handleRetry() {
    if (lastFailedMessage) {
      handleSend(lastFailedMessage);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-white">
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/95 backdrop-blur px-6 py-3.5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
            C
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-800">Chat Assistant</h1>
            <p className="text-xs text-slate-400">
              {sessionId ? `Session active - ${messages.length} messages` : "New conversation"}
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto">
          {messages.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center text-center text-slate-400 mt-20">
              <p className="text-sm">Send a message to start the conversation.</p>
            </div>
          )}

          {messages.map((msg, i) => {
            // Don't render the trailing empty assistant placeholder bubble
            // while still waiting for the first chunk -- the typing
            // indicator below covers that "waiting" state instead, so the
            // user doesn't see a redundant, empty message bubble AND the
            // three dots at the same time.
            const isEmptyPlaceholder =
              isLoading && i === messages.length - 1 && msg.role === "assistant" && msg.content === "";
            if (isEmptyPlaceholder) return null;

            return (
              <ChatMessage key={i} role={msg.role} content={msg.content} timestamp={msg.timestamp} />
            );
          })}

          {isLoading && <TypingIndicator />}

          <div ref={scrollAnchorRef} />
        </div>
      </main>

      <ErrorBanner error={error} onDismiss={() => setError(null)} onRetry={handleRetry} />

      <MessageInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
