import { create } from 'zustand';

interface UIState {
  sidebarOpen: boolean;
  selectedPeriod: 'full' | '1st' | '2nd';

  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setSelectedPeriod: (period: 'full' | '1st' | '2nd') => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  selectedPeriod: 'full',

  toggleSidebar: () => set(s => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setSelectedPeriod: (period) => set({ selectedPeriod: period }),
}));
