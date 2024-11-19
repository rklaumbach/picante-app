// frontend/src/pages/FullViewPage.tsx

'use client'

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';


interface Image {
  _id: string;
  imageUrl: string;
  filename: string;
  bodyPrompt: string;
  facePrompt: string;
  resolution: string;
  timestamp: string;
}

const FullViewPage: React.FC = () => {
  const router = useRouter();
  const [image, setImage] = useState<Image | null>(null);

  useEffect(() => {
    const storedImage = sessionStorage.getItem('fullViewImage');
    if (storedImage) {
      setImage(JSON.parse(storedImage));
      // Optionally, remove it from storage
      sessionStorage.removeItem('fullViewImage');
    } else {
      // If no image data is present, redirect to gallery
      router.push('/gallery');
    }
  }, [router.push]);

  if (!image) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <p className="text-lg text-gray-800">Loading...</p>
      </div>
    ); // Or a loading spinner
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 relative">
      {/* Close Button */}
      <button
        onClick={() => router.back}
        className="absolute top-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition duration-200"
        aria-label="Close Full View"
      >
        Close
      </button>

      {/* Image and Details Container */}
      <div className="app-container flex flex-col items-center w-full max-w-4xl">
        {/* Full Resolution Image */}
        <img
          src={image.imageUrl}
          alt={image.filename}
          className="w-full h-auto object-contain rounded-lg"
        />

        {/* Image Details */}
        <div className="mt-4 text-center text-gray-800">
          <h2 className="text-3xl font-bold">{image.filename}</h2>
          <p className="mt-2">
            <span className="font-semibold">Timestamp:</span>{' '}
            {new Date(image.timestamp).toLocaleString()}
          </p>
          <p className="mt-1">
            <span className="font-semibold">Body Prompt:</span> {image.bodyPrompt}
          </p>
          <p className="mt-1">
            <span className="font-semibold">Face Prompt:</span> {image.facePrompt}
          </p>
        </div>
      </div>
    </div>
  );
};

export default FullViewPage;
