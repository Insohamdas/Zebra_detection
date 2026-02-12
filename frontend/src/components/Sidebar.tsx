import { NavLink } from 'react-router-dom';
import { LayoutDashboard, ScanLine, PawPrint, Camera, Settings } from 'lucide-react';
import clsx from 'clsx';

const NAV_ITEMS = [
  { icon: LayoutDashboard, label: 'Dashboard', to: '/' },
  { icon: ScanLine, label: 'Identify', to: '/identify' },
  { icon: PawPrint, label: 'Zebras', to: '/zebras' },
  { icon: Camera, label: 'Cameras', to: '/cameras' },
  { icon: Settings, label: 'Settings', to: '/settings' },
];

export function Sidebar() {
  return (
    <aside className="hidden md:flex w-64 flex-col bg-slate-850 text-white h-screen fixed left-0 top-0 border-r border-slate-800">
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center">
          <span className="font-bold text-white text-lg">Z</span>
        </div>
        <div>
          <h1 className="font-bold text-lg tracking-tight">Zebra ID</h1>
          <p className="text-xs text-slate-400">System v1.0</p>
        </div>
      </div>

      <nav className="flex-1 px-4 py-4 space-y-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              )
            }
          >
            <item.icon size={20} />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
