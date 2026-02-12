import { useEffect, useState } from 'react';
import { Users, Eye, Camera, Activity, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { StatCard } from '../components/StatCard';
import { api } from '../api/mock';
import type { DashboardStats, Zebra } from '../types';

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentZebras, setRecentZebras] = useState<Zebra[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      const [statsData, zebrasData] = await Promise.all([
        api.getStats(),
        api.getZebras()
      ]);
      setStats(statsData);
      setRecentZebras(zebrasData.slice(0, 5));
      setLoading(false);
    };
    loadData();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-96 text-gray-500">Loading dashboard...</div>;
  }

  if (!stats) return null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500">Overview of the Zebra ID System</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Zebras"
          value={stats.totalZebras}
          icon={Users}
          trend="+12%"
          trendUp={true}
        />
        <StatCard
          title="New Today"
          value={stats.newToday}
          icon={Activity}
          trend="+2"
          trendUp={true}
        />
        <StatCard
          title="Total Sightings"
          value={stats.totalSightings.toLocaleString()}
          icon={Eye}
          trend="+5%"
          trendUp={true}
        />
        <StatCard
          title="Active Cameras"
          value={stats.activeCameras}
          icon={Camera}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="lg:col-span-2 card p-6">
          <h3 className="text-lg font-semibold mb-6">Sightings Activity</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.sightingsHistory}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{fill: '#6B7280'}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#6B7280'}} />
                <Tooltip
                  cursor={{fill: '#F3F4F6'}}
                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                />
                <Bar dataKey="count" fill="#22c55e" radius={[4, 4, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">Recent Sightings</h3>
            <Link to="/zebras" className="text-sm text-primary-600 hover:text-primary-700 font-medium flex items-center">
              View all <ArrowRight size={16} className="ml-1" />
            </Link>
          </div>
          <div className="space-y-4">
            {recentZebras.map((zebra) => (
              <div key={zebra.id} className="flex items-center gap-4 p-3 hover:bg-gray-50 rounded-lg transition-colors border border-transparent hover:border-gray-100">
                <img
                  src={zebra.thumbnailUrl}
                  alt={zebra.name}
                  className="w-12 h-12 rounded-lg object-cover bg-gray-200"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">{zebra.name}</p>
                  <p className="text-xs text-gray-500 truncate">
                    Seen {new Date(zebra.lastSeen).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                  </p>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                  zebra.status === 'new'
                    ? 'bg-blue-50 text-blue-700 border border-blue-100'
                    : 'bg-gray-100 text-gray-600 border border-gray-200'
                }`}>
                  {zebra.status === 'new' ? 'New' : 'Known'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
