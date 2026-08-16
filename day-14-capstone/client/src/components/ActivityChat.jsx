/**
 * ActivityChat.jsx
 * -------------------
 * The shared chat interface for all three activities (requirement #1:
 * "dedicated chat interface, which shall include a Back button"). The
 * SAME component handles all three -- differences (Brain Buster's
 * hint/give-up buttons, each activity's accent color) are driven by the
 * `activity` prop rather than three separate near-duplicate components.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { startSession, sendTurn, endSession, ApiError } from "../api/capstoneApi";
import { useInactivityTimer } from "../lib/useInactivityTimer";

const ACTIVITY_META = {
  brain_buster: { name: "Brain Buster", emoji: "🧩", color: "buster" },
  quick_fire: { name: "Quick Fire", emoji: "⚡", color: "fire" },
  ask_explore: { name: "Ask & Explore", emoji: "🔭", color: "explore" },
};

const COLOR_CLASSES = {
  buster: { header: "bg-buster", soft: "bg-buster-soft", text: "text-buster", ring: "focus:ring-buster/30", border: "focus:border-buster" },
  fire: { header: "bg-fire", soft: "bg-fire-soft", text: "text-fire", ring: "focus:ring-fire/30", border: "focus:border-fire" },
  explore: { header: "bg-explore", soft: "bg-explore-soft", text: "text-explore", ring: "focus:ring-explore/30", border: "focus:border-explore" },
};

export default function ActivityChat({ activity, onBack }) {
  const meta = ACTIVITY_META[activity];
  const colors = COLOR_CLASSES[meta.color];

  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hintsUsed, setHintsUsed] = useState(0);

  const scrollAnchorRef = useRef(null);
  const sessionIdRef = useRef(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleInactivityTimeout = useCallback(() => {
    endSession(sessionIdRef.current);
    onBack();
  }, [onBack]);

  useInactivityTimer(handleInactivityTimeout, true);

  useEffect(() => {
    let cancelled = false;
    setMessages([{ role: "model", content: "" }]);

    startSession(activity, {
      onChunk: (text) => {
        if (cancelled) return;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...updated[updated.length - 1], content: updated[updated.length - 1].content + text };
          return updated;
        });
      },
      onDone: () => {
        if (!cancelled) setIsLoading(false);
      },
    })
      .then((sid) => {
        if (!cancelled) {
          setSessionId(sid);
          sessionIdRef.current = sid;
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Something went wrong starting this activity.");
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activity]);

  async function handleBack() {
    await endSession(sessionId);
    onBack();
  }

  async function runTurn(message, action) {
    if (!sessionId) return;
    setError(null);
    setIsLoading(true);

    if (message) {
      setMessages((prev) => [...prev, { role: "user", content: message }]);
    }
    setMessages((prev) => [...prev, { role: "model", content: "" }]);

    try {
      await sendTurn(sessionId, message, action, {
        onChunk: (text) => {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...updated[updated.length - 1], content: updated[updated.length - 1].content + text };
            return updated;
          });
        },
        onNewItem: () => {
          setMessages((prev) => [...prev, { role: "model", content: "" }]);
        },
        onDone: () => setIsLoading(false),
      });

      if (action === "guess" || action === "answer") {
        setHintsUsed(0);
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        onBack();
        return;
      }
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setIsLoading(false);
    }
  }

  function handleSend() {
    const text = inputValue.trim();
    if (!text || isLoading) return;
    setInputValue("");
    const action = activity === "brain_buster" ? "guess" : activity === "quick_fire" ? "answer" : "ask";
    runTurn(text, action);
  }

  function handleHint() {
    if (isLoading || hintsUsed >= 3) return;
    setHintsUsed((h) => h + 1);
    runTurn("", "hint");
  }

  function handleGiveUp() {
    if (isLoading) return;
    setHintsUsed(0);
    runTurn("", "give_up");
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-bg">
      <header className={`${colors.header} sticky top-0 z-10 flex items-center gap-4 px-6 py-4 shadow-sm`}>
        <button
          onClick={handleBack}
          className="font-display font-bold text-white/90 hover:text-white flex items-center gap-1"
        >
          ← Back
        </button>
        <h1 className="font-display text-xl font-bold text-white flex items-center gap-2">
          <span aria-hidden="true">{meta.emoji}</span> {meta.name}
        </h1>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 font-body text-[15px] leading-relaxed ${
                  msg.role === "user"
                    ? `${colors.header} text-white rounded-br-sm`
                    : `${colors.soft} text-ink rounded-bl-sm`
                }`}
              >
                {msg.role === "user" ? (
                  msg.content
                ) : (
                  <ReactMarkdown>{msg.content || "\u00A0"}</ReactMarkdown>
                )}
              </div>
            </div>
          ))}
          <div ref={scrollAnchorRef} />
        </div>
      </main>

      {error && (
        <div className="mx-4 mb-2 max-w-2xl md:mx-auto w-full">
          <div className="rounded-xl bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700 font-body">
            {error}
          </div>
        </div>
      )}

      <div className="border-t border-black/5 bg-white px-4 py-4">
        <div className="max-w-2xl mx-auto">
          {activity === "brain_buster" && (
            <div className="flex gap-2 mb-3">
              <button
                onClick={handleHint}
                disabled={isLoading || hintsUsed >= 3}
                className={`font-display text-sm font-bold rounded-full px-4 py-2 border-2 border-buster ${colors.text}
                            hover:bg-buster-soft disabled:opacity-40 disabled:cursor-not-allowed transition-colors`}
              >
                💡 Hint ({3 - hintsUsed} left)
              </button>
              <button
                onClick={handleGiveUp}
                disabled={isLoading}
                className="font-display text-sm font-bold rounded-full px-4 py-2 border-2 border-muted/40 text-muted
                           hover:bg-black/5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                🏳️ Give Up
              </button>
            </div>
          )}

          <div className="flex items-end gap-3">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={1}
              placeholder={
                activity === "brain_buster" ? "Type your guess..."
                : activity === "quick_fire" ? "Type your answer..."
                : "Ask me anything..."
              }
              className={`flex-1 resize-none rounded-2xl border-2 border-black/10 px-4 py-3 font-body text-[15px]
                          text-ink placeholder:text-muted focus:outline-none focus:ring-4 ${colors.ring} ${colors.border}
                          disabled:opacity-50 disabled:cursor-not-allowed`}
              style={{ minHeight: "50px", maxHeight: "140px" }}
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !inputValue.trim()}
              className={`${colors.header} font-display font-bold text-white rounded-2xl px-6 py-3.5
                          disabled:opacity-40 disabled:cursor-not-allowed transition-opacity`}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
