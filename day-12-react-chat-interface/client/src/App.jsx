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
import { sendMessage, ChatApiError } from "./api/chatApi";

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
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const data = await sendMessage(text, sessionId);

      if (!sessionId) {
        setSessionId(data.session_id);
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response, timestamp: new Date().toISOString() },
      ]);
      setLastFailedMessage(null);
    } catch (err) {
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

          {messages.map((msg, i) => (
            <ChatMessage key={i} role={msg.role} content={msg.content} timestamp={msg.timestamp} />
          ))}

          {isLoading && <TypingIndicator />}

          <div ref={scrollAnchorRef} />
        </div>
      </main>

      <ErrorBanner error={error} onDismiss={() => setError(null)} onRetry={handleRetry} />

      <MessageInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
