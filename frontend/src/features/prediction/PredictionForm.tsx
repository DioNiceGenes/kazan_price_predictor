import { useMemo, useState } from 'react'
import { ApartmentMap } from './ApartmentMap'
import { PredictionResult } from './PredictionResult'
import { useUploadImagesMutation, usePredictPriceMutation } from './predictionApi'

import {
  calculateDistanceKm,
  findNearestMetro,
  KAZAN_CENTER,
} from '../../shared/constants/coordinates.ts'

export const PredictionForm = () => {
  const [uploadImages, { isLoading: isUploading }] = useUploadImagesMutation()
  const [predictPrice, { data, isLoading, error }] =
    usePredictPriceMutation()

  const [coordinates, setCoordinates] = useState({
    lat: 55.796127,
    lng: 49.106414,
  })

  const [images, setImages] = useState<File[]>([])
  const [imagePreviews, setImagePreviews] = useState<string[]>([])
  const [uploadError, setUploadError] = useState<string | null>(null)

  const [form, setForm] = useState({
    total_square: 58.5,
    rooms: 2,
    floor: 9,
    floors: 16,

    living_square: 32,
    kitchen_square: 9.5,

    balcony: 'есть',
    bathroom_type: 'совмещенный',
    sell_type: 'свободная продажа',

    is_new_building: false,
    is_studio: false,

    description:
      'Просторная квартира с евроремонтом.',
  })

  const metroInfo = useMemo(() => {
    return findNearestMetro(
      coordinates.lat,
      coordinates.lng,
    )
  }, [coordinates])

  const distToCenter = useMemo(() => {
    return calculateDistanceKm(
      coordinates.lat,
      coordinates.lng,
      KAZAN_CENTER.lat,
      KAZAN_CENTER.lng,
    )
  }, [coordinates])

  const handleChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
    >,
  ) => {
    const { name, value, type } = e.target

    setForm((prev) => ({
      ...prev,
      [name]:
        type === 'checkbox'
          ? (e.target as HTMLInputElement).checked
          : type === 'number'
            ? Number(value)
            : value,
    }))
  }

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []).slice(0, 3)
    setImages(files)
    
    // Создаем превью для каждого файла
    const previews: string[] = []
    files.forEach((file) => {
      const reader = new FileReader()
      reader.onload = (event) => {
        if (event.target?.result) {
          previews.push(event.target.result as string)
          if (previews.length === files.length) {
            setImagePreviews(previews)
          }
        }
      }
      reader.readAsDataURL(file)
    })
  }

  const removeImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index))
    setImagePreviews((prev) => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async (
    e: React.FormEvent,
  ) => {
    e.preventDefault()
    setUploadError(null)

    let imagePaths: string[] = []

    // Загружаем фотографии, если они есть
    if (images.length > 0) {
      try {
        const formData = new FormData()
        images.forEach((file) => {
          formData.append('files', file)
        })

        const uploadResult = await uploadImages(formData).unwrap()
        imagePaths = uploadResult.files.map((f) => f.saved_path)
      } catch (err) {
        setUploadError(
          'Ошибка при загрузке фотографий. Попробуйте снова.'
        )
        console.error('Upload error:', err)
        return
      }
    }

    // Отправляем запрос на предсказание с путями к фотографиям
    await predictPrice({
      ...form,

      metro: metroInfo.metro,
      dist_to_metro: Number(
        metroInfo.distance.toFixed(2),
      ),
      dist_to_center: Number(
        distToCenter.toFixed(2),
      ),

      image_paths: imagePaths,
    })
  }

  return (
    <div>
      <div className="mb-10">

        <h1 className="mt-5 text-6xl font-bold tracking-tight text-slate-900">
          Оценка квартиры
          <span className="block text-blue-600">
            в Казани
          </span>
        </h1>

        <p className="mt-5 max-w-2xl text-lg text-slate-600">
          Современная ML-модель оценки недвижимости
          с учетом локации, описания и параметров
          квартиры.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]"
      >
        {/* LEFT */}
        <div className="rounded-lg border border-slate-200 bg-white p-8 shadow-sm">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-slate-900">
                Параметры квартиры
              </h2>

              <p className="mt-1 text-sm text-slate-600">
                Заполните характеристики жилья
              </p>
            </div>

          </div>

          <div className="grid grid-cols-2 gap-5">
            <Input
              label="Общая площадь"
              name="total_square"
              value={form.total_square}
              onChange={handleChange}
            />

            <Input
              label="Комнаты"
              name="rooms"
              value={form.rooms}
              onChange={handleChange}
            />

            <Input
              label="Этаж"
              name="floor"
              value={form.floor}
              onChange={handleChange}
            />

            <Input
              label="Этажей"
              name="floors"
              value={form.floors}
              onChange={handleChange}
            />

            <Input
              label="Жилая площадь"
              name="living_square"
              value={form.living_square}
              onChange={handleChange}
            />

            <Input
              label="Кухня"
              name="kitchen_square"
              value={form.kitchen_square}
              onChange={handleChange}
            />
          </div>

          <div className="mt-6 grid gap-5">
            <Select
              label="Балкон"
              name="balcony"
              value={form.balcony}
              options={['есть', 'нет']}
              onChange={handleChange}
            />

            <Select
              label="Санузел"
              name="bathroom_type"
              value={form.bathroom_type}
              options={[
                'совмещенный',
                'раздельный',
              ]}
              onChange={handleChange}
            />

            <Select
              label="Тип продажи"
              name="sell_type"
              value={form.sell_type}
              options={[
                'свободная продажа',
                'альтернатива',
              ]}
              onChange={handleChange}
            />
          </div>

          <div className="mt-7 flex gap-8">
            <Checkbox
              label="Новостройка"
              name="is_new_building"
              checked={form.is_new_building}
              onChange={handleChange}
            />

            <Checkbox
              label="Студия"
              name="is_studio"
              checked={form.is_studio}
              onChange={handleChange}
            />
          </div>

          <div className="mt-7">
            <label className="mb-3 block text-sm font-medium text-slate-900">
              Описание квартиры
            </label>

            <textarea
              name="description"
              value={form.description}
              onChange={handleChange}
              rows={6}
              className="w-full rounded-md border border-slate-200 bg-white p-3 text-slate-900 outline-none transition-colors placeholder:text-slate-500 focus:border-slate-400 focus:ring-1 focus:ring-slate-400"
            />
          </div>

          <div className="mt-7">
            <label className="mb-3 block text-sm font-medium text-slate-900">
              Фотографии
            </label>

            {imagePreviews.length > 0 && (
              <div className="mb-4 grid grid-cols-3 gap-3">
                {imagePreviews.map((preview, index) => (
                  <div key={index} className="relative group">
                    <img
                      src={preview}
                      alt={`Preview ${index + 1}`}
                      className="h-20 w-20 rounded-md border border-slate-200 object-cover"
                    />
                    <button
                      type="button"
                      onClick={() => removeImage(index)}
                      className="absolute -top-2 -right-2 hidden group-hover:flex items-center justify-center h-6 w-6 rounded-full bg-red-500 text-white text-xs font-bold hover:bg-red-600 transition-colors"
                    >
                      ×
                    </button>
                    <p className="mt-1 text-xs text-slate-600 truncate">
                      {images[index]?.name || `Фото ${index + 1}`}
                    </p>
                  </div>
                ))}
              </div>
            )}

            <label className="relative flex cursor-pointer items-center justify-center rounded-md border-2 border-dashed border-slate-300 bg-slate-50 p-8 transition-colors hover:border-slate-400 hover:bg-slate-100">
              <div className="text-center">
                <svg
                  className="mx-auto h-8 w-8 text-slate-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                <p className="mt-2 text-sm font-medium text-slate-900">
                  Загрузить фото
                </p>
                <p className="text-xs text-slate-600">
                  {images.length}/3 изображений
                </p>
              </div>
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={handleImageChange}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {/* RIGHT */}
        <div className="space-y-6">
          <div className="overflow-hidden rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5">
              <h2 className="text-2xl font-semibold text-slate-900">
                Локация
              </h2>

              <p className="mt-1 text-sm text-slate-600">
                Выберите точку на карте
              </p>
            </div>

            <ApartmentMap
              position={[
                coordinates.lat,
                coordinates.lng,
              ]}
              onChange={(lat, lng) =>
                setCoordinates({ lat, lng })
              }
            />

            <div className="mt-5 grid gap-3">
              <InfoCard
                title="Метро"
                value={metroInfo.metro}
              />

              <InfoCard
                title="До метро"
                value={`${metroInfo.distance.toFixed(2)} км`}
              />

              <InfoCard
                title="До центра"
                value={`${distToCenter.toFixed(2)} км`}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || isUploading}
            className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isUploading
              ? 'Загрузка фото...'
              : isLoading
              ? 'Оценка...'
              : 'Оценить квартиру'}
          </button>

          {uploadError && (
            <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {uploadError}
            </div>
          )}

          {error && (
            <div className="rounded-md border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
              Ошибка запроса
            </div>
          )}

          {data && (
            <PredictionResult
              prediction={data.prediction}
            />
          )}
        </div>
      </form>
    </div>
  )
}

interface InputProps {
  label: string
  name: string
  value: number
  onChange: (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => void
}

const Input = ({
  label,
  name,
  value,
  onChange,
}: InputProps) => (
  <div>
    <label className="mb-2 block text-sm font-medium text-slate-900">
      {label}
    </label>

    <input
      type="number"
      name={name}
      value={value}
      onChange={onChange}
      className="h-9 w-full rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-500 focus:border-slate-400 focus:ring-1 focus:ring-slate-400"
    />
  </div>
)

interface SelectProps {
  label: string
  name: string
  value: string
  options: string[]
  onChange: (
    e: React.ChangeEvent<HTMLSelectElement>,
  ) => void
}

const Select = ({
  label,
  name,
  value,
  options,
  onChange,
}: SelectProps) => (
  <div>
    <label className="mb-2 block text-sm font-medium text-slate-900">
      {label}
    </label>

    <select
      name={name}
      value={value}
      onChange={onChange}
      className="h-9 w-full rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 outline-none transition-colors focus:border-slate-400 focus:ring-1 focus:ring-slate-400"
    >
      {options.map((option) => (
        <option
          key={option}
          value={option}
        >
          {option}
        </option>
      ))}
    </select>
  </div>
)

interface CheckboxProps {
  label: string
  name: string
  checked: boolean
  onChange: (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => void
}

const Checkbox = ({
  label,
  name,
  checked,
  onChange,
}: CheckboxProps) => (
  <label className="flex items-center gap-3 text-sm text-slate-700">
    <input
      type="checkbox"
      name={name}
      checked={checked}
      onChange={onChange}
      className="h-4 w-4 rounded border-slate-300 bg-white accent-blue-600"
    />

    {label}
  </label>
)

const InfoCard = ({
  title,
  value,
}: {
  title: string
  value: string
}) => (
  <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
    <p className="text-xs font-medium text-slate-600 uppercase tracking-wide">
      {title}
    </p>

    <p className="mt-1 text-sm font-semibold text-slate-900">
      {value}
    </p>
  </div>
)