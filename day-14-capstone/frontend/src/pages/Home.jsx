import { ACTIVITIES } from '../activities.js'

export default function HomeScreen({ onSelectActivity }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-brand-50 to-indigo-100 flex flex-col items-center justify-center p-6">
      <h1 className="text-4xl font-extrabold text-brand-700 mb-2 text-center">
        🌟 Learning Adventures
      </h1>
      <p className="text-slate-600 mb-10 text-center max-w-md">
        Pick an activity to get started!
      </p>

      <div className="grid gap-5 w-full max-w-xl">
        {ACTIVITIES.map((activity) => (
          <button
            key={activity.id}
            onClick={() => onSelectActivity(activity.id)}
            className={`bg-gradient-to-r ${activity.color} text-white rounded-2xl p-6 text-left shadow-lg hover:scale-[1.02] active:scale-[0.98] transition-transform`}
          >
            <div className="flex items-center gap-4">
              <span className="text-4xl">{activity.emoji}</span>
              <div>
                <div className="text-xl font-bold">{activity.name}</div>
                <div className="text-white/90 text-sm mt-1">{activity.description}</div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
