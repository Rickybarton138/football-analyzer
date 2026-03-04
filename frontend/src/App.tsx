import { Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import MatchLayout from './layouts/MatchLayout';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';
import NotFoundPage from './pages/NotFoundPage';
import LiveCoachingPage from './pages/LiveCoachingPage';
import OverviewPage from './pages/match/OverviewPage';
import CoachingPage from './pages/match/CoachingPage';
import TrainingPage from './pages/match/TrainingPage';
import TacticalPage from './pages/match/TacticalPage';
import AnalyticsPage from './pages/match/AnalyticsPage';
import PlayersPage from './pages/match/PlayersPage';

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="live" element={<LiveCoachingPage />} />
        <Route path="match/:id" element={<MatchLayout />}>
          <Route index element={<Navigate to="overview" replace />} />
          <Route path="overview" element={<OverviewPage />} />
          <Route path="coaching" element={<CoachingPage />} />
          <Route path="training" element={<TrainingPage />} />
          <Route path="tactical" element={<TacticalPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="players" element={<PlayersPage />} />
        </Route>
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
