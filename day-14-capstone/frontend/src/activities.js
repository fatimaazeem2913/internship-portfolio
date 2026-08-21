export const ACTIVITIES = [
  {
    id: 'brain_buster',
    name: 'Brain Buster',
    emoji: '🧩',
    description: 'Solve fun riddles! Get up to 3 hints if you get stuck.',
    color: 'from-purple-400 to-purple-600',
  },
  {
    id: 'quick_fire',
    name: 'Quick Fire',
    emoji: '⚡',
    description: 'Fast quiz questions across science, space, animals, and more!',
    color: 'from-amber-400 to-orange-500',
  },
  {
    id: 'ask_explore',
    name: 'Ask & Explore',
    emoji: '🔭',
    description: 'Ask anything you are curious about and explore the answer.',
    color: 'from-sky-400 to-blue-600',
  },
]

export function getActivityById(id) {
  return ACTIVITIES.find((a) => a.id === id)
}
