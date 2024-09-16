// src/components/LoginDialog.tsx

import React, { useState } from 'react';
import Dialog from './Dialog';
import Button from './Button';
import { useNavigate } from 'react-router-dom';
import SignUpDialog from './SignUpDialog';
import { useAuth } from '../contexts/AuthContext';

interface LoginDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const LoginDialog: React.FC<LoginDialogProps> = ({ isOpen, onClose }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUpOpen, setIsSignUpOpen] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleLogin = () => {
    // Placeholder for authentication logic
    // For now, simulate successful login
    login('dummy_token');
    onClose();
    navigate('/generate');
  };

  const openSignUp = () => {
    setIsSignUpOpen(true);
  };

  const closeSignUp = () => {
    setIsSignUpOpen(false);
  };

  return (
    <>
      <Dialog isOpen={isOpen} onClose={onClose}>
        <h2 className="text-2xl mb-4">Log In</h2>
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
              placeholder="Enter your password"
            />
          </div>
          <div className="flex justify-between">
            <Button
              text="Back"
              className="bg-gray-500 text-white px-4 py-2"
              onClick={onClose}
            />
            <div>
              <Button
                text="Log In"
                className="bg-blue-500 text-white px-4 py-2 mr-2"
                onClick={handleLogin}
              />
              <Button
                text="Sign Up"
                className="bg-green-500 text-white px-4 py-2"
                onClick={openSignUp}
              />
            </div>
          </div>
        </form>
      </Dialog>
      <SignUpDialog isOpen={isSignUpOpen} onClose={closeSignUp} />
    </>
  );
};

export default LoginDialog;
