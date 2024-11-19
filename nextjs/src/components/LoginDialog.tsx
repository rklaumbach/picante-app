// src/components/LoginDialog.tsx

'use client';

import React, { useState } from 'react';
import Dialog from './Dialog';
import Button from './Button';
import { signIn } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { toast } from 'react-toastify';
import Link from 'next/link'; // Import Link

interface LoginDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const LoginDialog: React.FC<LoginDialogProps> = ({ isOpen, onClose }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter();

  const handleLogin = async () => {
    const res = await signIn('credentials', {
      redirect: false,
      email,
      password,
    });

    if (res?.error) {
      if (res.error.includes('confirm your email')) {
        toast.error('Please confirm your email before logging in.');
      } else {
        toast.error(res.error);
      }
    } else {
      onClose();
      toast.success('Logged in successfully!');
      router.push('/generate');
    }
  };

  return (
    <Dialog isOpen={isOpen} onClose={onClose}>
      <h2 className="text-2xl mb-4">Log In</h2>
      <form onSubmit={(e) => e.preventDefault()}>
        {/* Email and Password Inputs */}
        <div className="mb-4">
          <label className="block text-left">Email:</label>
          <input
            type="email"
            className="w-full px-4 py-2 mb-2 bg-gray-100 rounded-lg"
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
            className="w-full px-4 py-2 mb-2 bg-gray-100 rounded-lg"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            placeholder="Enter your password"
          />
        </div>
        {/* "Forgot Password?" Link */}
        <div className="mb-4 text-right">
          <Link href="/forgot-password" className="text-blue-500 hover:underline">
            Forgot Password?
          </Link>
        </div>
        {/* Action Buttons */}
        <div className="flex justify-between">
          <Button
            text="Back"
            className="bg-gray-500 text-white px-4 py-2"
            onClick={onClose}
          />
          <Button
            text="Log In"
            className="bg-blue-500 text-white px-4 py-2"
            onClick={handleLogin}
          />
        </div>
      </form>
    </Dialog>
  );
};

export default LoginDialog;
