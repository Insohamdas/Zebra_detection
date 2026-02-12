import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Identify from './pages/Identify';
import Zebras from './pages/Zebras';
import ZebraDetail from './pages/ZebraDetail';
import Settings from './pages/Settings';
import Cameras from './pages/Cameras';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />

        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="identify" element={<Identify />} />
          <Route path="zebras" element={<Zebras />} />
          <Route path="zebras/:id" element={<ZebraDetail />} />
          <Route path="cameras" element={<Cameras />} />
          <Route path="settings" element={<Settings />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
