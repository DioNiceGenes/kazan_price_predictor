import { PredictionForm } from '../features/prediction/PredictionForm'

export const HomePage = () => {
  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <PredictionForm />
      </div>
    </div>
  )
}