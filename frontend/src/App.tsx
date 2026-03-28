import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MatchUpload from './pages/MatchUpload';
import MatchView from './pages/MatchView';
import Search from './pages/Search';
import Squad from './pages/Squad';

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<MatchUpload />} />
            <Route path="/match/:id" element={<MatchView />} />
            <Route path="/search" element={<Search />} />
            <Route path="/squad" element={<Squad />} />
            <Route path="*" element={
              <div className="text-center py-20">
                <h1 className="text-4xl font-bold text-zinc-400 mb-2">404</h1>
                <p className="text-zinc-500">Page not found</p>
              </div>
            } />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
