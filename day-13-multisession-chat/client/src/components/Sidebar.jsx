/**
 * Sidebar.jsx
 * --------------
 * Lists all active chat sessions. Clicking a session loads its full
 * message history (by setting it as the active session in App.jsx's
 * state). Includes the "New Chat" button.
 */

export default function Sidebar({ sessions, activeSessionId, onSelectSession, onNewChat }) {
  const sortedSessions = [...sessions].sort(
    (a, b) => new Date(b.lastActiveAt || b.createdAt) - new Date(a.lastActiveAt || a.createdAt)
  );

  return (
    <aside className="flex flex-col w-64 flex-shrink-0 h-screen bg-slate-50 border-r border-slate-200">
      <div className="p-3 border-b border-slate-200">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 text-white
                     text-sm font-medium py-2.5 hover:bg-blue-700 active:bg-blue-800 transition-colors"
        >
          <span className="text-lg leading-none">+</span>
          New Chat
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto py-2">
        {sortedSessions.length === 0 && (
          <p className="text-xs text-slate-400 text-center mt-6 px-4">
            No conversations yet. Click "New Chat" to start.
          </p>
        )}

        {sortedSessions.map((session) => {
          const isActive = session.id === activeSessionId;
          const displayTitle = session.title || "New conversation";
          const messageCount = session.messages.filter((m) => m.role !== "system").length;

          return (
            <button
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`w-full text-left px-3 py-2.5 mx-2 mb-1 rounded-lg text-sm transition-colors
                          ${isActive ? "bg-blue-100 text-blue-900" : "text-slate-700 hover:bg-slate-200"}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate font-medium">{displayTitle}</span>
                {!session.title && messageCount > 0 && (
                  <span
                    className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse"
                    title="Generating title..."
                  />
                )}
              </div>
              {messageCount > 0 && (
                <span className="text-xs text-slate-400">{messageCount} messages</span>
              )}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
