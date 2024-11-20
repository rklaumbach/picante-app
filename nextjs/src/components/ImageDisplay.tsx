// src/components/ImageDisplay.tsx

import React from 'react';
import Image from 'next/image';

interface ImageDisplayProps {
  imageUrl: string;
  width: number;
  height: number;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ imageUrl, width, height }) => {
  return (
    <div className="w-full mt-6">
      <Image
        src={imageUrl}
        alt="Generated"
        width={width}
        height={height}
        className="w-full h-auto rounded-lg"
      />
    </div>
  );
};

export default ImageDisplay;
