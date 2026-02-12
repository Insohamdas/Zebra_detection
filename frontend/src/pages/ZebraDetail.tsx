import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, MapPin, Calendar, Camera, Edit2, Trash2, GitMerge } from 'lucide-react';
import { api } from '../api/mock';
import type { Zebra, Sighting } from '../types';

export default function ZebraDetail() {
  const { id } = useParams<{ id: string }>();
  const [zebra, setZebra] = useState<Zebra | null>(null);
  const [sightings, setSightings] = useState<Sighting[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      if (!id) return;
      setLoading(true);
      const [zebraData, sightingsData] = await Promise.all([
        api.getZebra(id),
        api.getSightings(id)
      ]);
      if (zebraData) setZebra(zebraData);
      setSightings(sightingsData);
      setLoading(false);
    };
    loadData();
  }, [id]);

  if (loading) return <div className="p-8 text-center text-gray-500">Loading profile...</div>;
  if (!zebra) return <div className="p-8 text-center text-red-500">Zebra not found</div>;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <Link to="/zebras" className="inline-flex items-center text-sm text-gray-500 hover:text-gray-900 mb-4">
          <ArrowLeft size={16} className="mr-1" /> Back to Database
        </Link>
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
          <div className="flex items-start gap-6">
            <img
              src={zebra.thumbnailUrl}
              alt={zebra.name}
              className="w-32 h-32 rounded-xl object-cover border-4 border-white shadow-sm bg-gray-200"
            />
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-3xl font-bold text-gray-900">{zebra.name}</h1>
                <span className={`px-2.5 py-0.5 rounded-full text-sm font-medium ${
                  zebra.status === 'new' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                }`}>
                  {zebra.status === 'new' ? 'New' : 'Known'}
                </span>
              </div>
              <div className="flex flex-wrap gap-4 text-sm text-gray-500">
                <div className="flex items-center gap-1">
                  <Calendar size={16} />
                  First seen: {new Date(zebra.firstSeen).toLocaleDateString()}
                </div>
                <div className="flex items-center gap-1">
                  <MapPin size={16} />
                  Last location: {sightings[0]?.location || 'Unknown'}
                </div>
                <div className="flex items-center gap-1">
                  <Camera size={16} />
                  {zebra.sightingsCount} sightings
                </div>
              </div>
              <div className="mt-4 flex gap-2">
                {zebra.tags?.map(tag => (
                  <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs font-medium">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            <button className="btn btn-secondary">
              <Edit2 size={16} className="mr-2" /> Edit Notes
            </button>
            <button className="btn btn-secondary">
              <GitMerge size={16} className="mr-2" /> Merge
            </button>
            <button className="btn btn-danger bg-red-50 text-red-600 border-red-100 hover:bg-red-100 hover:border-red-200">
              <Trash2 size={16} />
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content - Timeline */}
        <div className="lg:col-span-2 space-y-6">
          <h2 className="text-lg font-semibold text-gray-900">Sighting History</h2>
          <div className="space-y-4">
            {sightings.map((sighting) => (
              <div key={sighting.id} className="card p-4 flex gap-4 hover:shadow-md transition-shadow">
                <img
                  src={sighting.imageUrl}
                  alt="Sighting"
                  className="w-32 h-24 rounded-lg object-cover bg-gray-100"
                />
                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-medium text-gray-900">{sighting.location}</h4>
                      <p className="text-sm text-gray-500">
                        {new Date(sighting.timestamp).toLocaleDateString()} at {new Date(sighting.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                    <span className="text-xs font-medium text-green-600 bg-green-50 px-2 py-1 rounded">
                      {(sighting.confidence * 100).toFixed(0)}% Match
                    </span>
                  </div>
                  <div className="mt-4 flex gap-2">
                    <button className="text-xs text-primary-600 hover:text-primary-700 font-medium">
                      View Full Image
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar - Notes & Map */}
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Notes</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              {zebra.notes || "No notes added for this zebra yet."}
            </p>
            <button className="mt-4 text-sm text-primary-600 hover:text-primary-700 font-medium">
              + Add Note
            </button>
          </div>

          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Typical Range</h3>
            <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center text-gray-400 text-sm">
              Map Placeholder
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
