/**
 * ChatMessage.jsx
 * ------------------
 * Light theme. User messages: amber-filled bubble, right-aligned.
 * Assistant messages: white/paper bubble with a thin amber left rule,
 * left-aligned -- same visual language as the sidebar's accent bar.
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
    <div className={`group flex w-full ${isUser ? "justify-end" : "justify-start"} mb-5`}>
      <div className={`flex flex-col max-w-[70%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`px-4 py-3 text-[14px] leading-relaxed break-words font-sans ${
            isUser
              ? "bg-accent text-accent-ink rounded-lg rounded-br-sm whitespace-pre-wrap"
              : "bg-white text-ink rounded-lg rounded-bl-sm border border-border border-l-2 border-l-accent/60 shadow-sm"
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
                strong: ({ children }) => <strong className="font-semibold text-accent">{children}</strong>,
                code: ({ children }) => (
                  <code className="bg-surface text-accent px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
                ),
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-2">
                    {children}
                  </a>
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          )}
        </div>

        <div className="flex items-center gap-3 mt-1.5 px-1">
          {timestamp && (
            <span className="font-mono text-[10px] text-muted tracking-wide">
              {new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}

          <div className="flex items-center gap-2.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleCopy}
              className="font-mono text-[10px] uppercase tracking-wide text-muted hover:text-accent"
              title="Copy message"
            >
              {copied ? "Copied" : "Copy"}
            </button>

            {!isUser && isLastAssistantMessage && (
              <button
                onClick={onRegenerate}
                disabled={isRegenerating}
                className="font-mono text-[10px] uppercase tracking-wide text-muted hover:text-accent disabled:opacity-50"
                title="Regenerate response"
              >
                {isRegenerating ? "Working..." : "Regenerate"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
