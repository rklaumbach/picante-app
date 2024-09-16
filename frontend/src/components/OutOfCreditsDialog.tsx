// src/components/OutOfCreditsDialog.tsx

import React from 'react';
import Dialog from './Dialog';
import Button from './Button';
import { useNavigate } from 'react-router-dom';

interface OutOfCreditsDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const OutOfCreditsDialog: React.FC<OutOfCreditsDialogProps> = ({ isOpen, onClose }) => {
  const navigate = useNavigate();

  return (
    <Dialog isOpen={isOpen} onClose={onClose}>
      <h2 className="text-2xl mb-4">Out of Free Image Credits!</h2>
      <p className="mb-4">
        Free users of Picante can only generate up to 5 images per day. To get full access to our app, subscribe to
        Picante Pro. Try free for 30 days, then only $10/mo. (LIMITED TIME OFFER! Only $5/mo for the first 3 months.)
        Premium users also receive exclusive perks like:
      </p>
      <ul className="list-disc list-inside mb-4">
        <li>Unlimited Images</li>
        <li>Reduced Waiting Time (requests prioritized)</li>
        <li>AI upscaling (2-4x resolution increase)</li>
        <li>Exclusive access to animation features (coming soon)</li>
      </ul>
      <Button
        text="Subscribe Now"
        className="bg-red-500 text-white"
        onClick={() => {
          onClose();
          navigate('/subscribe'); // Assuming you have a subscription page
        }}
      />
    </Dialog>
  );
};

export default OutOfCreditsDialog;
