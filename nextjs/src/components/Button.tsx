// src/components/Button.tsx

import React from 'react';

interface ButtonProps {
  text: string;
  className?: string;
  onClick: () => void;
  disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({ text, className, onClick, disabled }) => (
  <button
    className={className}
    onClick={onClick}
    disabled={disabled}
    type="button" // Ensure the button type is specified
  >
    {text}
  </button>
);

export default Button;
