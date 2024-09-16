// src/pages/FullViewPage.tsx

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

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
  const navigate = useNavigate();
  const [image, setImage] = useState<Image | null>(null);

  useEffect(() => {
    const storedImage = sessionStorage.getItem('fullViewImage');
    if (storedImage) {
      setImage(JSON.parse(storedImage));
      // Optionally, remove it from storage
      sessionStorage.removeItem('fullViewImage');
    } else {
      // If no image data is present, redirect to gallery
      navigate('/gallery');
    }
  }, [navigate]);

  if (!image) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
        <p className="text-lg">Loading...</p>
      </div>
    ); // Or a loading spinner
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white relative">
      {/* Close Button */}
      <button
        onClick={() => navigate(-1)}
        className="absolute top-4 right-4 bg-red-600 px-4 py-2 rounded-lg hover:bg-red-700 transition duration-200"
        aria-label="Close Full View"
      >
        Close
      </button>

      {/* Full Resolution Image */}
      <img
        src={image.imageUrl}
        alt={image.filename}
        className="max-w-full max-h-full rounded-lg"
      />

      {/* Image Details */}
      <div className="mt-4 text-center">
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
  );
};

export default FullViewPage;
