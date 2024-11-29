// src/components/CachedImage.tsx

import Image from 'next/image';
import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { ImageData } from '@/types/types';

interface CachedImageProps {
  imageData: ImageData;
}

const CachedImage: React.FC<CachedImageProps> = ({ imageData }) => {
  const [currentSrc, setCurrentSrc] = useState(imageData.image_url);
  const [retryCount, setRetryCount] = useState(0);

  const handleError = async () => {
    if (retryCount >= 3) {
      toast.error('Failed to load image after multiple attempts.');
      return;
    }

    try {
      const response = await fetch('/api/images/refresh-signed-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageId: imageData.id }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.newSignedUrl) {
          setCurrentSrc(data.newSignedUrl);
          setRetryCount((prev) => prev + 1);
        } else {
          throw new Error('No new signed URL returned.');
        }
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to refresh signed URL.');
      }
    } catch (error: any) {
      console.error('Error refreshing signed URL:', error);
      toast.error('Failed to load image. Please try again later.');
    }
  };

  return (
    <Image
      src={currentSrc}
      alt={imageData.filename}
      width={imageData.width}
      height={imageData.height}
      onError={handleError}
      loading="lazy"
      unoptimized={true}
      className="rounded-lg transform transition-transform duration-200 hover:scale-105"
    />
  );
};

export default CachedImage;
