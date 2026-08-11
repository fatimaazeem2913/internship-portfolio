/**
 * ChatMessage.jsx
 * ------------------
 * Renders a single message bubble, with genuinely distinct visual
 * treatment for user vs. assistant messages -- not just a label change,
 * but different alignment, background color, and a visual role
 * indicator, so a user can tell who said what at a glance while
 * scrolling, without reading every word.
 */

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
            className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap break-words ${
              isUser
                ? "bg-blue-600 text-white rounded-br-sm"
                : "bg-slate-100 text-slate-800 rounded-bl-sm"
            }`}
          >
            {content}
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
