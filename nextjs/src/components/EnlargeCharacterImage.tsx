// src/components/EnlargeCharacterImage.tsx

'use client';

import React from 'react';
import { Character } from '@/types/types';

interface EnlargeCharacterImageProps {
  character: Character;
  isEnlarged: boolean;
  onClose: () => void;
}

const EnlargeCharacterImage: React.FC<EnlargeCharacterImageProps> = ({ character, isEnlarged, onClose }) => {
  if (!isEnlarged) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg p-4 relative w-1/2"
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside
      >
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-gray-700 hover:text-gray-900 text-2xl"
          aria-label="Close"
        >
          &times;
        </button>
        <img src={character.signed_image_url} alt={character.name} className="w-full h-auto object-contain" />
      </div>
    </div>
  );
};

export default EnlargeCharacterImage;
