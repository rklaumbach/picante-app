// src/components/Button.tsx

import React from 'react';

interface ButtonProps {
  text: string;
  className?: string;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

const Button: React.FC<ButtonProps> = ({ text, className, onClick, type = 'button' }) => {
  return (
    <button
      type={type}
      className={`flex justify-center items-center px-4 py-2 rounded-lg ${className} hover:opacity-90 transition duration-200`}
      onClick={onClick}
    >
      {text}
    </button>
  );
};

export default Button;
