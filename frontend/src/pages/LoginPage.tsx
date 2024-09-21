// src/pages/LoginPage.tsx

import React, { useState } from 'react';
import Header from '../components/Header';
import Button from '../components/Button';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  const handleLogin = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (response.ok) {
        login(data.token);
        navigate('/generate');
      } else {
        alert(data.error || 'Login failed.');
      }
    } catch (error) {
      console.error('Login error:', error);
      alert('An error occurred during login.');
    }
  };

  return (
    <main className="flex flex-col items-center px-7 pb-10 mx-auto w-full max-w-md bg-red-500 text-white min-h-screen">
      <Header title="Login" />

      <form className="w-full mt-8" onSubmit={(e) => e.preventDefault()}>
        <div className="mt-4">
          <label className="text-white">Email:</label>
          <input
            type="email"
            className="w-full mt-2 px-4 py-2 bg-white rounded-lg text-black"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        <div className="mt-4">
          <label className="text-white">Password:</label>
          <input
            type="password"
            className="w-full mt-2 px-4 py-2 bg-white rounded-lg text-black"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        <Button
          text="Log In"
          className="mt-6 bg-zinc-800 text-white"
          onClick={handleLogin}
        />
      </form>
    </main>
  );
};

export default LoginPage;
