// frontend/src/App.tsx

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// Import your pages
import GalleryPage from './pages/GalleryPage';
import FullViewPage from './pages/FullViewPage';
import GenerateImagePage from './pages/GenerateImagePage';
import LandingPage from './pages/LandingPage';
import AccountSettingsPage from './pages/AccountSettingsPage';
import LoginPage from './pages/LoginPage';
//import SubscribePage from './pages/SubscribePage'; // Create this page

// Import PrivateRoute
import PrivateRoute from './components/PrivateRoute';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        {/* Landing Page */}
        <Route path="/" element={<LandingPage />} />

        {/* Authentication Routes */}
        <Route path="/login" element={<LoginPage />} />
        {/* <Route path="/signup" element={<SignUpPage />} /> */}

        {/* Protected Routes */}
        <Route
          path="/generate"
          element={
            <PrivateRoute>
              <GenerateImagePage />
            </PrivateRoute>
          }
        />
        <Route
          path="/account"
          element={
            <PrivateRoute>
              <AccountSettingsPage />
            </PrivateRoute>
          }
        />
        {/* <Route
          path="/subscribe"
          element={
            <PrivateRoute>
              <SubscribePage />
            </PrivateRoute>
          }
        /> */}

        {/* Gallery and Full View Routes */}
        <Route path="/gallery" element={<GalleryPage />} />
        <Route path="/full-view" element={<FullViewPage />} />

        {/* Catch-All Route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default App;
