// src/components/PromptInput.tsx

import React from 'react';

interface PromptInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
}

const PromptInput: React.FC<PromptInputProps> = ({ label, value, onChange }) => {
  return (
    <div className="w-full mt-4">
      <label className="text-white">{label}:</label>
      <textarea
        className="w-full mt-2 px-4 py-2 bg-white rounded-lg text-black"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
      />
    </div>
  );
};

export default PromptInput;
