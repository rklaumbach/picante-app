// src/components/SignUpDialog.tsx

import React, { useState } from 'react';
import Dialog from './Dialog';
import Button from './Button';
import { useAuth } from '../contexts/AuthContext';

interface SignUpDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const SignUpDialog: React.FC<SignUpDialogProps> = ({ isOpen, onClose }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [accountCreated, setAccountCreated] = useState(false);
  const { login } = useAuth();

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  const handleSignUp = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();

      if (response.ok) {
        login(data.token);
        setAccountCreated(true);
      } else {
        alert(data.error || 'Sign up failed.');
      }
    } catch (error) {
      console.error('Sign up error:', error);
      alert('An error occurred during sign up.');
    }
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
              <label className="block text-left">Email:</label>
              <input
                type="email"
                className="w-full mt-2 px-4 py-2 bg-gray-100 rounded-lg"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="Enter your email"
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
