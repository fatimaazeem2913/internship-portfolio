/**
 * TypingIndicator.jsx
 * -----------------------
 * Three animated dots shown while a request is in flight -- gives the
 * user something to look at while waiting (perceived responsiveness,
 * not actual speed -- see Day 8's streaming findings).
 */

export default function TypingIndicator() {
  return (
    <div className="flex w-full justify-start mb-5" aria-live="polite" aria-label="Composing a reply">
      <div className="px-4 py-3.5 rounded-lg rounded-bl-sm bg-white border border-border border-l-2 border-l-accent/60 shadow-sm flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-accent/70 animate-bounce [animation-delay:-0.3s]" />
        <span className="w-1.5 h-1.5 rounded-full bg-accent/70 animate-bounce [animation-delay:-0.15s]" />
        <span className="w-1.5 h-1.5 rounded-full bg-accent/70 animate-bounce" />
      </div>
    </div>
  );
}
