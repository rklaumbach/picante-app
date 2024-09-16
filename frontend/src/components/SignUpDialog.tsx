// src/components/SignUpDialog.tsx

import React, { useState } from 'react';
import Dialog from './Dialog';
import Button from './Button';

interface SignUpDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const SignUpDialog: React.FC<SignUpDialogProps> = ({ isOpen, onClose }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [accountCreated, setAccountCreated] = useState(false);

  const handleSignUp = () => {
    // Placeholder for sign-up logic
    // For now, simulate account creation
    setAccountCreated(true);
  };

  const handleClose = () => {
    setAccountCreated(false);
    onClose();
  };

  return (
    <Dialog isOpen={isOpen} onClose={handleClose}>
      {!accountCreated ? (
        <>
          <h2 className="text-2xl mb-4">Sign Up</h2>
          <form onSubmit={(e) => e.preventDefault()}>
            <div className="mb-4">
              <label className="block text-left">Username:</label>
              <input
                type="text"
                className="w-full mt-2 px-4 py-2 bg-gray-100 rounded-lg"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                placeholder="Choose a username"
              />
            </div>
            <div className="mb-4">
              <label className="block text-left">Password:</label>
              <input
                type="password"
                className="w-full mt-2 px-4 py-2 bg-gray-100 rounded-lg"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="Create a password"
              />
            </div>
            <div className="flex justify-between">
              <Button
                text="Back"
                className="bg-gray-500 text-white px-4 py-2"
                onClick={handleClose}
              />
              <Button
                text="Sign Up"
                className="bg-green-500 text-white px-4 py-2"
                onClick={handleSignUp}
              />
            </div>
          </form>
        </>
      ) : (
        <>
          <h2 className="text-2xl mb-4">Account Created!</h2>
          <p className="mb-4">Your account has been successfully created.</p>
          <div className="flex justify-end">
            <Button
              text="Close"
              className="bg-blue-500 text-white px-4 py-2"
              onClick={handleClose}
            />
          </div>
        </>
      )}
    </Dialog>
  );
};

export default SignUpDialog;
