export interface ApartmentInput {
  total_square: number
  rooms: number
  floor: number
  floors: number

  living_square?: number
  kitchen_square?: number
  dist_to_metro?: number
  dist_to_center?: number

  balcony: string
  bathroom_type: string
  sell_type: string
  metro?: string

  is_new_building: boolean
  is_studio: boolean

  description?: string
  image_paths?: string[]
}

export interface Prediction {
  recommended_price: number
  lower_bound: number
  upper_bound: number
  confidence_interval: string
  model_used: string
  processing_time_ms: number
}

export interface PredictionResponse {
  status: string
  prediction: Prediction
}

export interface UploadedFile {
  filename: string
  saved_path: string
  unique_id: string
}

export interface UploadImagesResponse {
  status: string
  files: UploadedFile[]
  count: number
}