// src/components/UnsubscribeModal.tsx

import React, { useState } from 'react';
import Dialog from './Dialog';
import Button from './Button';

interface UnsubscribeModalProps {
  isOpen: boolean;
  onClose: () => void;
  userEmail: string;
  onUnsubscribeSuccess: () => void;
}

const UnsubscribeModal: React.FC<UnsubscribeModalProps> = ({
  isOpen,
  onClose,
  userEmail,
  onUnsubscribeSuccess,
}) => {
  const [reasons, setReasons] = useState<string[]>([]);
  const [additionalFeedback, setAdditionalFeedback] = useState('');

  const unsubscribeReasons = [
    'Too expensive',
    'Not using the service enough',
    'Found a better alternative',
    'Privacy concerns',
    'Other',
  ];

  const handleReasonChange = (reason: string) => {
    setReasons((prevReasons) =>
      prevReasons.includes(reason)
        ? prevReasons.filter((r) => r !== reason)
        : [...prevReasons, reason]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const response = await fetch('/api/unsubscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({
          reasons,
          additionalFeedback,
          email: userEmail,
        }),
      });

      if (response.ok) {
        alert('Your subscription has been cancelled. We have sent a confirmation to your email.');
        onUnsubscribeSuccess();
        onClose();
      } else {
        alert('There was an error processing your request. Please try again later.');
      }
    } catch (error) {
      console.error('Error during unsubscription:', error);
      alert('An unexpected error occurred. Please try again later.');
    }
  };

  return (
    <Dialog isOpen={isOpen} onClose={onClose}>
      <h2 className="text-2xl mb-4">We're sorry to see you go!</h2>
      <p>Please let us know why you're unsubscribing:</p>
      <form onSubmit={handleSubmit}>
        <div className="mt-4">
          {unsubscribeReasons.map((reason, index) => (
            <div key={index} className="flex items-center">
              <input
                type="checkbox"
                id={`reason-${index}`}
                checked={reasons.includes(reason)}
                onChange={() => handleReasonChange(reason)}
                className="mr-2"
              />
              <label htmlFor={`reason-${index}`}>{reason}</label>
            </div>
          ))}
        </div>

        <div className="mt-4">
          <label htmlFor="additionalFeedback">Additional Feedback (optional):</label>
          <textarea
            id="additionalFeedback"
            className="w-full mt-2 p-2 border rounded-lg"
            rows={3}
            value={additionalFeedback}
            onChange={(e) => setAdditionalFeedback(e.target.value)}
          />
        </div>

        <div className="flex justify-end mt-6">
          <Button text="Cancel" className="bg-gray-300 text-black mr-4" onClick={onClose} />
          <Button text="Submit" className="bg-red-600 text-white" type="submit" />
        </div>
      </form>
    </Dialog>
  );
};

export default UnsubscribeModal;
