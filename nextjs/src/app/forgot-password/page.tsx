// src/app/forgot-password/page.tsx

'use client';

import React, { useState } from 'react';
import Header from '../../components/Header';
import Button from '../../components/Button';
import { toast } from 'react-toastify';
import { useRouter } from 'next/navigation';

const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handlePasswordReset = async () => {
    try {
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('Password reset link sent! Please check your email.');
        setError(null);
        toast.success('Password reset link sent! Please check your email.');
      } else {
        setError(data.error || 'Failed to send password reset link.');
        setMessage(null);
        toast.error(data.error || 'Failed to send password reset link.');
      }
    } catch (err) {
      console.error('Error sending password reset link:', err);
      setError('An unexpected error occurred.');
      setMessage(null);
      toast.error('An unexpected error occurred.');
    }
  };

  return (
    <>
      <Header title="Forgot Password" />
      <main className="flex flex-col items-center px-4 pb-10 mx-auto w-full max-w-7xl min-h-screen pt-24">
        <div className="app-container flex flex-col items-center w-full">
          <h2 className="text-2xl text-white mb-4">Reset Your Password</h2>
          <p className="text-white mb-6">Enter your email address to receive a password reset link.</p>
          <div className="w-full max-w-md">
            <input
              type="email"
              className="w-full px-4 py-2 mb-4 bg-gray-200 rounded-lg text-black"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            {/* Optional Feedback Messages */}
            {message && <p className="text-green-500 mb-4">{message}</p>}
            {error && <p className="text-red-500 mb-4">{error}</p>}
            <Button
              text="Send Reset Link"
              className="w-full bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition duration-200"
              onClick={handlePasswordReset}
              disabled={!email.trim()}
            />
          </div>
        </div>
      </main>
    </>
  );
};

export default ForgotPasswordPage;
