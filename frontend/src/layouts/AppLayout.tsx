import { Link, Outlet, useLocation } from 'react-router-dom';
import { useUIStore } from '../stores/uiStore';
import {
  LayoutDashboard, Upload, Radio, Menu, X
} from 'lucide-react';
import { cn } from '../lib/utils';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/upload', label: 'New Match', icon: Upload },
  { path: '/live', label: 'Live Coaching', icon: Radio },
];

export default function AppLayout() {
  const location = useLocation();
  const { sidebarOpen, toggleSidebar } = useUIStore();

  return (
    <div className="min-h-screen bg-background flex">
      {/* Sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 bg-surface border-r border-border transition-all duration-300 flex flex-col shadow-sm',
          sidebarOpen ? 'w-56' : 'w-16'
        )}
      >
        {/* Logo */}
        <div className="h-16 flex items-center px-4 border-b border-border">
          <Link to="/" className="flex items-center gap-2 min-w-0">
            <div className="w-8 h-8 bg-gradient-to-br from-pitch-deep to-pitch rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-white text-sm font-bold">CM</span>
            </div>
            {sidebarOpen && (
              <div className="flex flex-col min-w-0">
                <span className="text-base font-bold text-text-primary leading-tight truncate">
                  <span className="text-pitch">Coach</span><span className="text-pitch-deep">Mentor</span>
                </span>
                <span className="text-[10px] text-text-muted leading-none">Match Analysis</span>
              </div>
            )}
          </Link>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 px-2 space-y-1">
          {NAV_ITEMS.map(item => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-btn text-sm font-medium transition-all',
                  isActive
                    ? 'bg-pitch-light text-pitch-deep'
                    : 'text-text-secondary hover:text-text-primary hover:bg-gray-100'
                )}
                title={!sidebarOpen ? item.label : undefined}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Toggle */}
        <button
          onClick={toggleSidebar}
          className="h-12 flex items-center justify-center border-t border-border text-text-muted hover:text-text-primary transition-colors"
        >
          {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </aside>

      {/* Main */}
      <div className={cn('flex-1 transition-all duration-300', sidebarOpen ? 'ml-56' : 'ml-16')}>
        <Outlet />
      </div>
    </div>
  );
}
