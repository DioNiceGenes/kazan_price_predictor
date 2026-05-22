import type { PricePrediction } from '../../shared/types/prediction'

interface Props {
  prediction: PricePrediction
}

export const PredictionResult = ({
  prediction,
}: Props) => {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900">
              Результат оценки
            </h2>

            <p className="mt-1 text-sm text-slate-600">
              ML Prediction
            </p>
          </div>

          <div className="rounded-full border border-slate-200 bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
            Success
          </div>
        </div>
      </div>

      <div className="p-8">
        <div className="rounded-md border border-slate-200 bg-slate-50 p-6">
          <p className="text-xs font-medium text-slate-600 uppercase tracking-wide">
            Рекомендуемая цена
          </p>

          <h3 className="mt-3 text-4xl font-bold tracking-tight text-slate-900">
            {prediction.recommended_price.toLocaleString(
              'ru-RU',
            )}{' '}
            ₽
          </h3>

          <div className="mt-8 grid gap-4 md:grid-cols-2">
            <PriceCard
              label="Нижняя граница"
              value={prediction.lower_bound}
            />

            <PriceCard
              label="Верхняя граница"
              value={prediction.upper_bound}
            />
          </div>
        </div>

        <div className="mt-5 grid gap-4">
          <MetaCard
            label="Диапазон"
            value={prediction.confidence_interval}
          />

          <MetaCard
            label="Модель"
            value={prediction.model_used}
          />

          <MetaCard
            label="Время обработки"
            value={`${prediction.processing_time_ms} ms`}
          />
        </div>
      </div>
    </div>
  )
}

const PriceCard = ({
  label,
  value,
}: {
  label: string
  value: number
}) => (
  <div className="rounded-md border border-slate-200 bg-white p-4">
    <p className="text-xs font-medium text-slate-600 uppercase tracking-wide">
      {label}
    </p>

    <p className="mt-2 text-xl font-bold text-slate-900">
      {value.toLocaleString('ru-RU')} ₽
    </p>
  </div>
)

const MetaCard = ({
  label,
  value,
}: {
  label: string
  value: string
}) => (
  <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
    <p className="text-xs font-medium text-slate-600 uppercase tracking-wide">
      {label}
    </p>

    <p className="mt-2 text-sm font-medium text-slate-900">
      {value}
    </p>
  </div>
)