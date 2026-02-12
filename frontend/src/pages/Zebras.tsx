import { useEffect, useState } from 'react';
import { Search, Filter, ChevronLeft, ChevronRight, Trash2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { api } from '../api/mock';
import type { Zebra } from '../types';

export default function Zebras() {
  const [zebras, setZebras] = useState<Zebra[]>([]);
  const [filteredZebras, setFilteredZebras] = useState<Zebra[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [locationFilter, setLocationFilter] = useState('all');
  const [sortBy, setSortBy] = useState<'recent' | 'oldest' | 'sightings'>('recent');

  const loadZebras = async () => {
    setLoading(true);
    const data = await api.getZebras(statusFilter === 'all' ? undefined : statusFilter);
    setZebras(data);
    setLoading(false);
  };

  const handleClearAll = async () => {
    if (confirm(`Are you sure you want to delete ALL ${zebras.length} zebras from the database? This action cannot be undone.`)) {
      setLoading(true);
      await api.clearAllZebras();
      await loadZebras();
    }
  };

  useEffect(() => {
    loadZebras();
  }, [statusFilter]);

  useEffect(() => {
    let result = [...zebras];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(z =>
        z.id.toLowerCase().includes(query) ||
        z.name.toLowerCase().includes(query) ||
        z.notes?.toLowerCase().includes(query) ||
        z.tags?.some(tag => tag.toLowerCase().includes(query))
      );
    }

    // Location filter
    if (locationFilter !== 'all') {
      result = result.filter(z => z.tags?.includes(locationFilter));
    }

    // Sorting
    if (sortBy === 'recent') {
      result.sort((a, b) => new Date(b.lastSeen).getTime() - new Date(a.lastSeen).getTime());
    } else if (sortBy === 'oldest') {
      result.sort((a, b) => new Date(a.firstSeen).getTime() - new Date(b.firstSeen).getTime());
    } else if (sortBy === 'sightings') {
      result.sort((a, b) => b.sightingsCount - a.sightingsCount);
    }

    setFilteredZebras(result);
  }, [zebras, searchQuery, locationFilter, sortBy]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Zebra Database</h1>
          <p className="text-gray-500">Manage and track identified individuals</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn btn-secondary">
            <Filter size={16} className="mr-2" />
            Filter
          </button>
          <button className="btn btn-primary">
            Export CSV
          </button>
          <button
            onClick={handleClearAll}
            disabled={zebras.length === 0}
            className="btn btn-danger"
          >
            <Trash2 size={16} className="mr-2" />
            Clear All
          </button>
        </div>
      </div>

      <div className="card overflow-hidden">
        {/* Toolbar */}
        <div className="p-4 border-b border-gray-200 bg-gray-50 space-y-4">
          <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
            <div className="relative w-full sm:w-96">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
              <input
                type="text"
                placeholder="Search by ID, name, tags, or notes..."
                className="input pl-10 w-full"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2 w-full sm:w-auto flex-wrap">
              <select
                className="input"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all">All Status</option>
                <option value="new">New Only</option>
                <option value="known">Known Only</option>
              </select>

              <select
                className="input"
                value={locationFilter}
                onChange={(e) => setLocationFilter(e.target.value)}
              >
                <option value="all">All Locations</option>
                <option value="North Zone">North Zone</option>
                <option value="South Zone">South Zone</option>
                <option value="Upload">Upload</option>
              </select>

              <select
                className="input"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'recent' | 'oldest' | 'sightings')}
              >
                <option value="recent">Most Recent</option>
                <option value="oldest">Oldest First</option>
                <option value="sightings">Most Sightings</option>
              </select>
            </div>
          </div>

          {/* Results count */}
          <div className="flex items-center justify-between text-sm">
            <p className="text-gray-600">
              Showing <span className="font-semibold">{filteredZebras.length}</span> of <span className="font-semibold">{zebras.length}</span> zebras
            </p>
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="text-primary-600 hover:text-primary-700 font-medium"
              >
                Clear search
              </button>
            )}
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-gray-50 text-gray-500 font-medium border-b border-gray-200">
              <tr>
                <th className="px-6 py-3">Image</th>
                <th className="px-6 py-3">Zebra ID</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3">Last Seen</th>
                <th className="px-6 py-3">Sightings</th>
                <th className="px-6 py-3">Tags</th>
                <th className="px-6 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    Loading database...
                  </td>
                </tr>
              ) : zebras.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    No zebras found matching your criteria.
                  </td>
                </tr>
              ) : filteredZebras.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    <p className="font-medium">No zebras match your filters</p>
                    <p className="text-sm mt-1">Try adjusting your search or filter settings</p>
                  </td>
                </tr>
              ) : (
                filteredZebras.map((zebra) => (
                  <tr key={zebra.id} className="hover:bg-gray-50 transition-colors group">
                    <td className="px-6 py-4">
                      <img
                        src={zebra.thumbnailUrl}
                        alt={zebra.name}
                        className="w-20 h-20 rounded-lg object-cover bg-gray-200 border border-gray-300 shadow-sm"
                      />
                    </td>
                    <td className="px-6 py-4">
                      <div>
                        <div className="font-medium text-gray-900">{zebra.name}</div>
                        <div className="text-xs text-gray-500">{zebra.id}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        zebra.status === 'new'
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-green-100 text-green-800'
                      }`}>
                        {zebra.status === 'new' ? 'New' : 'Known'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-500">
                      {new Date(zebra.lastSeen).toLocaleDateString()}
                      <div className="text-xs text-gray-400">
                        {new Date(zebra.lastSeen).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-500">
                      {zebra.sightingsCount}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-1 flex-wrap">
                        {zebra.tags?.map(tag => (
                          <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link to={`/zebras/${zebra.id}`} className="btn btn-secondary py-1.5 px-3 text-xs">
                        View
                      </Link>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-500">
            Showing <span className="font-medium">1</span> to <span className="font-medium">{zebras.length}</span> of <span className="font-medium">{zebras.length}</span> results
          </div>
          <div className="flex gap-2">
            <button className="btn btn-secondary p-2 disabled:opacity-50" disabled>
              <ChevronLeft size={16} />
            </button>
            <button className="btn btn-secondary p-2 disabled:opacity-50" disabled>
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
