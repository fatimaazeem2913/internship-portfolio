/**
 * TypingIndicator.jsx
 * -----------------------
 * Three animated dots shown while a request is in flight -- the same
 * "give the user something to look at while waiting" principle Day 8
 * covered for streaming vs. non-streaming (perceived responsiveness, not
 * actual speed). This doesn't make the backend respond faster; it makes
 * the wait feel less like the app has frozen.
 */

export default function TypingIndicator() {
  return (
    <div className="flex w-full justify-start mb-4" aria-live="polite" aria-label="Assistant is typing">
      <div className="flex items-end gap-2">
        <div
          className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold text-white bg-slate-600"
          aria-hidden="true"
        >
          AI
        </div>
        <div className="px-4 py-3 rounded-2xl rounded-bl-sm bg-slate-100 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.3s]" />
          <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.15s]" />
          <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" />
        </div>
      </div>
    </div>
  );
}
