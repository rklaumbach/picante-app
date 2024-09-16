// src/components/BottomNav.tsx

import React from 'react';
import { useNavigate } from 'react-router-dom';

const BottomNav: React.FC = () => {
  const navigate = useNavigate();

  return (
    <nav className="fixed bottom-0 left-0 w-full bg-white shadow-lg flex justify-around items-center h-16 z-50">
      <button
        className="text-blue-600 hover:text-blue-800"
        onClick={() => navigate('/generate')}
      >
        Generate Image
      </button>
      <button
        className="text-blue-600 hover:text-blue-800"
        onClick={() => navigate('/gallery')}
      >
        Gallery
      </button>
      <button
        className="text-blue-600 hover:text-blue-800"
        onClick={() => navigate('/account')}
      >
        Account Settings
      </button>
    </nav>
  );
};

export default BottomNav;
