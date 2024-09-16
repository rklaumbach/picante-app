// src/components/ImageDisplay.tsx

import React from 'react';

interface ImageDisplayProps {
  imageUrl: string;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ imageUrl }) => {
  return (
    <div className="w-full mt-6">
      <img src={imageUrl} alt="Generated" className="w-full rounded-lg" />
    </div>
  );
};

export default ImageDisplay;
