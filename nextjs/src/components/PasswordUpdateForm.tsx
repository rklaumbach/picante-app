// src/components/PasswordUpdateForm.tsx

'use client';

import React, { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Button from './Button';
import { supabaseClient } from '../lib/supabaseClient';
import { toast } from 'react-toastify';

const PasswordUpdateForm: React.FC = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const accessToken = searchParams.get('access_token'); // Supabase uses 'access_token' as a query parameter
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken) {
      setError('Invalid or missing access token.');
      toast.error('Invalid or missing access token.');
    }
  }, [accessToken]);

  const handlePasswordUpdate = async () => {
    if (newPassword !== confirmNewPassword) {
      setError('Passwords do not match.');
      toast.error('Passwords do not match.');
      return;
    }

    try {
      const { data, error } = await supabaseClient.auth.updateUser({
        password: newPassword,
      });

      if (error) {
        console.error('Error updating password:', error);
        setError(error.message);
        toast.error(error.message);
      } else {
        setMessage('Password updated successfully!');
        setError(null);
        toast.success('Password updated successfully! Redirecting to login...');
        // Redirect to login after a short delay
        setTimeout(() => {
          router.push('/');
        }, 3000);
      }
    } catch (err: any) {
      console.error('Unexpected error updating password:', err);
      setError('An unexpected error occurred.');
      toast.error('An unexpected error occurred.');
    }
  };

  return (
    <main className="flex flex-col items-center px-4 pb-10 mx-auto w-full max-w-7xl min-h-screen pt-24">
      <div className="app-container flex flex-col items-center w-full">
        <h2 className="text-2xl text-white mb-4">Set Your New Password</h2>
        <div className="w-full max-w-md">
          <input
            type="password"
            className="w-full px-4 py-2 mb-4 bg-gray-200 rounded-lg text-black"
            placeholder="Enter new password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            required
          />
          <input
            type="password"
            className="w-full px-4 py-2 mb-4 bg-gray-200 rounded-lg text-black"
            placeholder="Confirm new password"
            value={confirmNewPassword}
            onChange={(e) => setConfirmNewPassword(e.target.value)}
            required
          />
          {/* Optional Feedback Messages */}
          {message && <p className="text-green-500 mb-4">{message}</p>}
          {error && <p className="text-red-500 mb-4">{error}</p>}
          <Button
            text="Update Password"
            className="w-full bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition duration-200"
            onClick={handlePasswordUpdate}
            disabled={!newPassword.trim() || !confirmNewPassword.trim()}
          />
        </div>
      </div>
    </main>
  );
};

export default PasswordUpdateForm;
