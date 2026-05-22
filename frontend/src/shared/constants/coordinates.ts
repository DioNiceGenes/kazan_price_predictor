export const KAZAN_CENTER = {
  lat: 55.796127,
  lng: 49.106414,
}

export const METROS = [
  {
    name: 'Кремлёвская',
    lat: 55.796289,
    lng: 49.108795,
  },
  {
    name: 'Площадь Тукая',
    lat: 55.786602,
    lng: 49.124294,
  },
  {
    name: 'Суконная слобода',
    lat: 55.775219,
    lng: 49.142282,
  },
  {
    name: 'Аметьево',
    lat: 55.765224,
    lng: 49.165684,
  },
  {
    name: 'Горки',
    lat: 55.760112,
    lng: 49.187424,
  },
  {
    name: 'Проспект Победы',
    lat: 55.749038,
    lng: 49.207896,
  },
  {
    name: 'Дубравная',
    lat: 55.742137,
    lng: 49.220504,
  },
]

const toRad = (value: number) => {
  return (value * Math.PI) / 180
}

export const calculateDistanceKm = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
) => {
  const R = 6371

  const dLat = toRad(lat2 - lat1)
  const dLon = toRad(lon2 - lon1)

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2)

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))

  return Number((R * c).toFixed(2))
}

export const findNearestMetro = (
  lat: number,
  lng: number,
) => {
  let nearest = METROS[0]
  let minDistance = Infinity

  for (const metro of METROS) {
    const distance = calculateDistanceKm(
      lat,
      lng,
      metro.lat,
      metro.lng,
    )

    if (distance < minDistance) {
      minDistance = distance
      nearest = metro
    }
  }

  return {
    metro: nearest.name,
    distance: minDistance,
  }
}


