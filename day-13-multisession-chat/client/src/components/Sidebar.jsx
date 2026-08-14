/**
 * Sidebar.jsx
 * --------------
 * Light theme. Each session card shows a thin amber accent bar on its
 * left edge when active or hovered -- the one carried-over signature
 * detail from the original design, re-tuned for a light background.
 */

export default function Sidebar({ sessions, activeSessionId, onSelectSession, onNewChat }) {
  const sortedSessions = [...sessions].sort(
    (a, b) => new Date(b.lastActiveAt || b.createdAt) - new Date(a.lastActiveAt || a.createdAt)
  );

  return (
    <aside className="flex flex-col w-72 flex-shrink-0 h-screen bg-surface border-r border-border">
      <div className="px-5 pt-6 pb-4">
        <h2 className="font-display text-lg text-ink tracking-tight">Chats</h2>
        <p className="font-mono text-[10px] text-muted uppercase tracking-widest mt-0.5">
          {sessions.length} conversation{sessions.length === 1 ? "" : "s"}
        </p>
      </div>

      <div className="px-3 pb-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 rounded border border-accent/50 bg-white text-accent
                     font-sans text-sm font-medium py-2.5 hover:bg-accent hover:text-accent-ink transition-colors"
        >
          <span className="text-base leading-none">+</span>
          New Chat
        </button>
      </div>

      <div className="mx-3 border-t border-border" />

      <nav className="flex-1 overflow-y-auto py-2">
        {sortedSessions.length === 0 && (
          <p className="font-sans text-xs text-muted text-center mt-8 px-6 leading-relaxed">
            No conversations yet. Click "New Chat" to start.
          </p>
        )}

        {sortedSessions.map((session) => {
          const isActive = session.id === activeSessionId;
          const displayTitle = session.title || "New chat";
          const messageCount = session.messages.filter((m) => m.role !== "system").length;

          return (
            <button
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`group relative w-full text-left pl-4 pr-3 py-3 mx-2 mb-1 rounded-sm transition-colors
                          ${isActive ? "bg-white shadow-sm" : "hover:bg-surface-hover"}`}
            >
              <span
                className={`absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-full transition-colors
                            ${isActive ? "bg-accent" : "bg-transparent group-hover:bg-accent/30"}`}
              />

              <div className="flex items-baseline justify-between gap-2">
                <span
                  className={`font-display text-[15px] truncate ${
                    isActive ? "text-ink" : "text-ink/75"
                  }`}
                >
                  {displayTitle}
                </span>
                {!session.title && messageCount > 0 && (
                  <span
                    className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-accent animate-pulse"
                    title="Naming this chat..."
                  />
                )}
              </div>
              {messageCount > 0 && (
                <span className="font-mono text-[10px] text-muted tracking-wide">
                  {messageCount} messages
                </span>
              )}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
