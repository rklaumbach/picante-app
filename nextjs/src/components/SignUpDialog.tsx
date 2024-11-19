// src/components/SignUpDialog.tsx

'use client';

import React, { useState } from 'react';
import Dialog from './Dialog';
import Button from './Button';
// Uncomment the following lines if you're using react-toastify
// import { toast } from 'react-toastify';
// import 'react-toastify/dist/ReactToastify.css';

interface SignUpDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const SignUpDialog: React.FC<SignUpDialogProps> = ({ isOpen, onClose }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSignUp = async () => {
    // Reset error and success messages
    setError(null);
    setSuccessMessage(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      // Uncomment the following line if using react-toastify
      // toast.error('Passwords do not match.');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        setSuccessMessage('Registration successful! Please check your email to confirm your address.');
        // Uncomment the following line if using react-toastify
        // toast.success('Registration successful! Please check your email to confirm your address.');
        // Optionally, close the dialog after a short delay
        setTimeout(() => {
          onClose();
          // Optionally, redirect to login page
          // router.push('/'); // Ensure you have access to router if you choose to redirect
        }, 3000);
        // Clear form fields
        setEmail('');
        setPassword('');
        setConfirmPassword('');
      } else {
        setError(data.error || 'Registration failed.');
        // Uncomment the following line if using react-toastify
        // toast.error(data.error || 'Registration failed.');
      }
    } catch (err) {
      console.error('Error during sign-up:', err);
      setError('An unexpected error occurred.');
      // Uncomment the following line if using react-toastify
      // toast.error('An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog isOpen={isOpen} onClose={onClose}>
      <h2 className="text-2xl mb-4">Sign Up</h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSignUp();
        }}
        className="flex flex-col space-y-4"
      >
        {/* Email Input */}
        <div>
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
        {/* Password Input */}
        <div>
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
        {/* Confirm Password Input */}
        <div>
          <label className="block text-left">Confirm Password:</label>
          <input
            type="password"
            className="w-full mt-2 px-4 py-2 bg-gray-100 rounded-lg"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            placeholder="Confirm your password"
          />
        </div>
        {/* Display Success or Error Messages */}
        {successMessage && (
          <p className="text-green-500 text-center">{successMessage}</p>
        )}
        {error && (
          <p className="text-red-500 text-center">{error}</p>
        )}
        {/* Action Buttons */}
        <div className="flex justify-between">
          <Button
            text="Back"
            className="bg-gray-500 text-white px-4 py-2"
            onClick={onClose}
            disabled={loading}
          />
          <Button
            text={loading ? 'Signing Up...' : 'Sign Up'}
            className="bg-green-500 text-white px-4 py-2"
            onClick={handleSignUp}
            disabled={loading}
          />
        </div>
      </form>
    </Dialog>
  );
};

export default SignUpDialog;
