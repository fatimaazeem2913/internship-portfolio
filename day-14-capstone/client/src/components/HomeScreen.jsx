/**
 * HomeScreen.jsx
 * -----------------
 * The landing page (requirement #1): three activity cards, each with
 * its own color identity carried through to that activity's chat screen.
 */

const ACTIVITIES = [
  {
    id: "brain_buster",
    name: "Brain Buster",
    emoji: "🧩",
    description: "Solve fun riddles! Get hints if you're stuck.",
    color: "buster",
  },
  {
    id: "quick_fire",
    name: "Quick Fire",
    emoji: "⚡",
    description: "Answer quick questions about science, space, animals & more!",
    color: "fire",
  },
  {
    id: "ask_explore",
    name: "Ask & Explore",
    emoji: "🔭",
    description: "Curious about something? Ask me anything!",
    color: "explore",
  },
];

const COLOR_CLASSES = {
  buster: {
    bg: "bg-buster-soft",
    text: "text-buster",
    button: "bg-buster hover:bg-buster/90",
    ring: "hover:ring-buster/30",
  },
  fire: {
    bg: "bg-fire-soft",
    text: "text-fire",
    button: "bg-fire hover:bg-fire/90",
    ring: "hover:ring-fire/30",
  },
  explore: {
    bg: "bg-explore-soft",
    text: "text-explore",
    button: "bg-explore hover:bg-explore/90",
    ring: "hover:ring-explore/30",
  },
};

export default function HomeScreen({ onSelectActivity }) {
  return (
    <div className="min-h-screen flex flex-col items-center px-6 py-16 bg-bg">
      <h1 className="font-display text-5xl font-extrabold text-ink text-center mb-3">
        Learning Adventures
      </h1>
      <p className="font-body text-lg text-muted text-center mb-14 max-w-md">
        Pick an activity to start playing and learning!
      </p>

      <div className="grid gap-6 w-full max-w-4xl md:grid-cols-3">
        {ACTIVITIES.map((activity) => {
          const colors = COLOR_CLASSES[activity.color];
          return (
            <button
              key={activity.id}
              onClick={() => onSelectActivity(activity.id)}
              className={`flex flex-col items-center text-center rounded-3xl p-8 ${colors.bg}
                          ring-4 ring-transparent ${colors.ring} transition-all hover:-translate-y-1
                          shadow-sm hover:shadow-md`}
            >
              <span className="text-6xl mb-4" aria-hidden="true">{activity.emoji}</span>
              <h2 className={`font-display text-2xl font-bold ${colors.text} mb-2`}>
                {activity.name}
              </h2>
              <p className="font-body text-sm text-ink/70 mb-6">{activity.description}</p>
              <span className={`font-display font-bold text-white ${colors.button} rounded-full px-6 py-2.5 transition-colors`}>
                Let's go!
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
