/**
 * ChatMessage.jsx
 * ------------------
 * Renders a single message bubble, with genuinely distinct visual
 * treatment for user vs. assistant messages -- alignment, background
 * color, and a visual role indicator.
 *
 * MARKDOWN RENDERING (fix): Gemini's replies are genuinely formatted
 * using markdown syntax (**bold**, numbered lists, etc.) -- without a
 * markdown renderer, React just displays that syntax as literal text
 * (you'd see the raw ** characters). react-markdown parses the text and
 * renders REAL bold/lists/etc. instead. User messages are NOT rendered
 * as markdown (plain text only) -- there's no reason to interpret a
 * human's own typed message as formatting syntax.
 */

import ReactMarkdown from "react-markdown";

export default function ChatMessage({ role, content, timestamp }) {
  const isUser = role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"} mb-4`}>
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
                : "bg-slate-100 text-slate-800 rounded-bl-sm markdown-body"
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
          {timestamp && (
            <span
              className={`text-[11px] text-slate-400 mt-1 px-1 ${
                isUser ? "text-right" : "text-left"
              }`}
            >
              {new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
