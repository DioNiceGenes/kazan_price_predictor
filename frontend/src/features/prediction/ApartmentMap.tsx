import {
  MapContainer,
  Marker,
  TileLayer,
  useMapEvents,
} from 'react-leaflet'

interface Props {
  position: [number, number]
  onChange: (lat: number, lng: number) => void
}

function MapClickHandler({ onChange }: Props) {
  useMapEvents({
    click(e) {
      onChange(e.latlng.lat, e.latlng.lng)
    },
  })

  return null
}

export const ApartmentMap = ({
  position,
  onChange,
}: Props) => {
  return (
    <MapContainer
      center={position}
      zoom={12}
      className="h-[400px] w-full rounded-2xl"
    >
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <Marker position={position} />

      <MapClickHandler
        position={position}
        onChange={onChange}
      />
    </MapContainer>
  )
}
