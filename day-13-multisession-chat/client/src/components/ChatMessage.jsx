/**
 * ChatMessage.jsx
 * ------------------
 * Renders a single message bubble, with genuinely distinct visual
 * treatment for user vs. assistant messages, PLUS (Day 13) per-message
 * actions: copy text to clipboard, and -- for the LAST assistant message
 * only -- a regenerate button.
 *
 * WHY REGENERATE IS ONLY SHOWN ON THE LAST ASSISTANT MESSAGE:
 * Regenerating an OLDER reply wouldn't make sense given how the backend
 * implements it (server/main.py's /regenerate endpoint pops the LAST
 * turn in the session's history, unconditionally) -- offering the button
 * on every message would let a user click it on a message that isn't
 * actually the one that gets regenerated, a genuinely confusing UX bug.
 * Restricting it to isLastAssistantMessage keeps the UI honest about
 * what the button will actually do.
 */

import { useState } from "react";
import ReactMarkdown from "react-markdown";

export default function ChatMessage({ role, content, timestamp, isLastAssistantMessage, onRegenerate, isRegenerating }) {
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.warn("Clipboard write failed:", e);
    }
  }

  return (
    <div className={`group flex w-full ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`flex items-end gap-2 max-w-[75%] ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        <div
          className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold text-white ${
            isUser ? "bg-blue-600" : "bg-slate-600"
          }`}
          aria-hidden="true"
        >
          {isUser ? "U" : "AI"}
        </div>

        <div className="flex flex-col">
          <div
            className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed break-words ${
              isUser
                ? "bg-blue-600 text-white rounded-br-sm whitespace-pre-wrap"
                : "bg-slate-100 text-slate-800 rounded-bl-sm"
            }`}
          >
            {isUser ? (
              content
            ) : (
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                  ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol>,
                  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                  code: ({ children }) => (
                    <code className="bg-slate-200 px-1 py-0.5 rounded text-xs font-mono">{children}</code>
                  ),
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">
                      {children}
                    </a>
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            )}
          </div>

          <div className={`flex items-center gap-2 mt-1 px-1 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
            {timestamp && (
              <span className="text-[11px] text-slate-400">
                {new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            )}

            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                className="text-[11px] text-slate-400 hover:text-slate-700 px-1"
                title="Copy message"
              >
                {copied ? "Copied!" : "Copy"}
              </button>

              {!isUser && isLastAssistantMessage && (
                <button
                  onClick={onRegenerate}
                  disabled={isRegenerating}
                  className="text-[11px] text-slate-400 hover:text-slate-700 px-1 disabled:opacity-50"
                  title="Regenerate response"
                >
                  {isRegenerating ? "Regenerating..." : "Regenerate"}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
