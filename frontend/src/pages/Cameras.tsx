import { useEffect, useState } from 'react';
import { Camera, Plus, MapPin, Clock, Signal, Video, Settings as SettingsIcon, Trash2, Edit2, Power } from 'lucide-react';

interface CameraDevice {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'error';
  lastSeen: string;
  detectionCount: number;
  fps: number;
  resolution: string;
  streamUrl?: string;
}

const MOCK_CAMERAS: CameraDevice[] = [
  {
    id: 'CAM-001',
    name: 'Waterhole Alpha',
    location: 'North Sector',
    status: 'offline',
    lastSeen: new Date(Date.now() - 7200000).toISOString(),
    detectionCount: 0,
    fps: 0,
    resolution: '1920x1080',
  },
  {
    id: 'CAM-002',
    name: 'River Crossing',
    location: 'East Sector',
    status: 'offline',
    lastSeen: new Date(Date.now() - 5400000).toISOString(),
    detectionCount: 0,
    fps: 0,
    resolution: '1280x720',
  },
];

export default function Cameras() {
  const [cameras, setCameras] = useState<CameraDevice[]>(MOCK_CAMERAS);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'offline':
        return 'text-gray-500 bg-gray-50 border-gray-200';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-500 bg-gray-50 border-gray-200';
    }
  };

  const getSignalStrength = (status: string, fps: number) => {
    if (status === 'offline') return 0;
    if (fps >= 25) return 100;
    if (fps >= 15) return 60;
    return 30;
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Camera Management</h1>
          <p className="text-gray-500">Monitor and manage detection cameras across all zones</p>
        </div>
        <button className="btn btn-primary">
          <Plus size={18} className="mr-2" />
          Add Camera
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase">Total Cameras</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{cameras.length}</p>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <Camera size={24} className="text-blue-600" />
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase">Online</p>
              <p className="text-2xl font-bold text-green-600 mt-1">
                {cameras.filter(c => c.status === 'online').length}
              </p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <Signal size={24} className="text-green-600" />
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase">Total Detections</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                {cameras.reduce((sum, c) => sum + c.detectionCount, 0)}
              </p>
            </div>
            <div className="p-3 bg-purple-50 rounded-lg">
              <Video size={24} className="text-purple-600" />
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase">Offline</p>
              <p className="text-2xl font-bold text-red-600 mt-1">
                {cameras.filter(c => c.status === 'offline').length}
              </p>
            </div>
            <div className="p-3 bg-red-50 rounded-lg">
              <Power size={24} className="text-red-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Camera Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {cameras.map((camera) => (
          <div key={camera.id} className="card overflow-hidden hover:shadow-lg transition-shadow duration-300">
            {/* Camera Preview */}
            <div className="aspect-video bg-gray-900 relative">
              {camera.status === 'online' ? (
                <div className="w-full h-full flex items-center justify-center">
                  <img
                    src={`https://picsum.photos/seed/${camera.id}/640/360`}
                    alt={camera.name}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-3 right-3 flex gap-2">
                    <span className="px-2 py-1 bg-red-600 text-white text-xs font-bold rounded flex items-center gap-1">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                      LIVE
                    </span>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
                  <Camera size={48} className="mb-2" />
                  <p className="text-sm">Camera Offline</p>
                </div>
              )}
            </div>

            {/* Camera Info */}
            <div className="p-4 space-y-4">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{camera.name}</h3>
                  <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                    <div className="flex items-center gap-1">
                      <MapPin size={14} />
                      {camera.location}
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock size={14} />
                      {new Date(camera.lastSeen).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                    </div>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(camera.status)}`}>
                  {camera.status.toUpperCase()}
                </span>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-3 pt-3 border-t border-gray-100">
                <div>
                  <p className="text-xs text-gray-500">Resolution</p>
                  <p className="text-sm font-semibold text-gray-900 mt-0.5">{camera.resolution}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">FPS</p>
                  <p className="text-sm font-semibold text-gray-900 mt-0.5">{camera.fps}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Detections</p>
                  <p className="text-sm font-semibold text-gray-900 mt-0.5">{camera.detectionCount}</p>
                </div>
              </div>

              {/* Signal Strength */}
              <div className="pt-3 border-t border-gray-100">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-500">Signal Strength</p>
                  <p className="text-xs font-semibold text-gray-700">{getSignalStrength(camera.status, camera.fps)}%</p>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      camera.status === 'online' ? 'bg-green-500' : 'bg-gray-400'
                    }`}
                    style={{ width: `${getSignalStrength(camera.status, camera.fps)}%` }}
                  ></div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-3">
                <button className="btn btn-primary flex-1 text-xs">
                  <Video size={14} className="mr-1" />
                  View Stream
                </button>
                <button className="btn btn-secondary p-2">
                  <SettingsIcon size={16} />
                </button>
                <button className="btn btn-secondary p-2">
                  <Edit2 size={16} />
                </button>
                <button className="btn btn-danger p-2">
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
