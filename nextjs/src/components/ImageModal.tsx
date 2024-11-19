// src/components/ImageModal.tsx

import React from 'react';
import { Image } from '@/types/types'; // Import the unified Image interface

interface ImageModalProps {
  image: Image;
  onClose: () => void;
  onDownload: () => void;
  onDelete: () => void;
  onFullView: () => void;
}

const ImageModal: React.FC<ImageModalProps> = ({ image, onClose, onDownload, onDelete, onFullView }) => {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 transition-opacity duration-300"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg overflow-hidden w-11/12 max-w-4xl transform transition-transform duration-300 scale-100 hover:scale-105"
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside the modal
      >
        {/* Close Button */}
        <div className="flex justify-end p-2">
          <button
            onClick={onClose}
            className="text-gray-700 hover:text-gray-800 text-xl font-bold"
            aria-label="Close"
          >
            ✖
          </button>
        </div>

        {/* Image and Details */}
        <div className="flex flex-col md:flex-row">
          {/* Image Section */}
          <div className="md:w-1/2 flex justify-center items-center p-4">
            <img
              src={image.image_url}
              alt={image.filename}
              className="w-full h-auto object-contain rounded-lg"
            />
          </div>

          {/* Details Section */}
          <div className="md:w-1/2 p-4 text-gray-800">
            <h3 id="modal-title" className="text-2xl font-semibold mb-2">{image.filename}</h3>
            <p className="mb-1">
              <span className="font-semibold">Timestamp:</span>{' '}
              {new Date(image.created_at).toLocaleString()}
            </p>
            <p className="mb-1">
              <span className="font-semibold">Body Prompt:</span> {image.body_prompt}
            </p>
            <p className="mb-4">
              <span className="font-semibold">Face Prompt:</span> {image.face_prompt}
            </p>

            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={onDownload}
                className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition duration-200"
              >
                Download
              </button>
              <button
                onClick={() => {
                  onDelete();
                  onClose();
                }}
                className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition duration-200"
              >
                Delete
              </button>
              <button
                onClick={onFullView}
                className="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600 transition duration-200"
              >
                Full View
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageModal;
