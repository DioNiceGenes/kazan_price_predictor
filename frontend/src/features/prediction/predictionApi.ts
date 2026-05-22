import { api } from '../../shared/api/api'
import type {
  ApartmentInput,
  PredictionResponse,
  UploadImagesResponse,
} from '../../shared/types/prediction'

export const predictionApi = api.injectEndpoints({
  endpoints: (builder) => ({
    uploadImages: builder.mutation<
      UploadImagesResponse,
      FormData
    >({
      query: (formData) => ({
        url: '/upload-images',
        method: 'POST',
        body: formData,
      }),
    }),
    predictPrice: builder.mutation<
      PredictionResponse,
      ApartmentInput
    >({
      query: (body) => ({
        url: '/predict',
        method: 'POST',
        body,
      }),
    }),
  }),
})

export const {
  useUploadImagesMutation,
  usePredictPriceMutation,
} = predictionApi