// src/components/Header.tsx

import React from 'react';

interface HeaderProps {
  title: string;
}

const Header: React.FC<HeaderProps> = ({ title }) => {
  return (
    <header className="fixed top-0 left-0 w-full h-16 bg-red-600 text-white flex items-center justify-center z-50">
      <h1 className="text-3xl font-bold">{title}</h1>
    </header>
  );
};

export default Header;
