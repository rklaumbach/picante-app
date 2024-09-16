// src/components/ImageDetailsDialog.tsx

import React from 'react';
import Dialog from './Dialog';
import Button from './Button';

interface ImageDetails {
  timestamp: string;
  resolution: string;
  bodyPrompt: string;
  facePrompt: string;
}

interface ImageDetailsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  imageDetails: ImageDetails;
}

const ImageDetailsDialog: React.FC<ImageDetailsDialogProps> = ({ isOpen, onClose, imageDetails }) => {
  const { timestamp, resolution, bodyPrompt, facePrompt } = imageDetails;

  return (
    <Dialog isOpen={isOpen} onClose={onClose}>
      <h2 className="text-2xl mb-4">Image Details</h2>
      <p>
        <strong>Timestamp:</strong> {timestamp}
      </p>
      <p>
        <strong>Resolution:</strong> {resolution}
      </p>
      <p>
        <strong>Body Prompt:</strong> {bodyPrompt}
      </p>
      <p>
        <strong>Face Prompt:</strong> {facePrompt}
      </p>
      <div className="flex justify-end mt-4">
        <Button text="Ok, Thanks!" className="bg-blue-500 text-white" onClick={onClose} />
      </div>
    </Dialog>
  );
};

export default ImageDetailsDialog;
