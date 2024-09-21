// src/pages/AccountSettingsPage.tsx

import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Button from '../components/Button';
import UnsubscribeModal from '../components/UnsubscribeModal';
import BottomNav from '../components/BottomNav';
import { useNavigate } from 'react-router-dom';

interface User {
  email: string;
  subscriptionStatus: string;
}

const AccountSettingsPage: React.FC = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState<User>({ email: '', subscriptionStatus: '' });
  const [email, setEmail] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [showUnsubscribeModal, setShowUnsubscribeModal] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  useEffect(() => {
    // Fetch user data
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          console.warn('No token found, redirecting to login.');
          navigate('/login');
          return;
        }

        const response = await fetch(`${API_BASE_URL}/api/user/profile`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const userData = await response.json();
          setUser(userData);
          setEmail(userData.email);
        } else {
          navigate('/login');
        }
      } catch (error) {
        console.error('Error fetching user data:', error);
      }
    };

    fetchUserData();
  }, [navigate, API_BASE_URL]);

  // Function to update user profile (email)
  const handleUpdateProfile = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        console.warn('No token found, redirecting to login.');
        navigate('/login');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/user/profile`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ email }),
      });

      if (response.ok) {
        const data = await response.json();
        alert('Email updated successfully!');
        setUser((prevUser) => ({ ...prevUser, email: data.email }));
      } else {
        const data = await response.json();
        alert(`Failed to update email: ${data.error || response.statusText}`);
      }
    } catch (error) {
      console.error('Error updating email:', error);
      alert('An error occurred while updating the email.');
    }
  };

  // Function to update password
  const handleUpdatePassword = async () => {
    // Client-side validation
    if (newPassword !== confirmNewPassword) {
      alert('New passwords do not match.');
      return;
    }

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        console.warn('No token found, redirecting to login.');
        navigate('/login');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/user/update-password`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ currentPassword, newPassword }),
      });

      if (response.ok) {
        alert('Password updated successfully!');
        setCurrentPassword('');
        setNewPassword('');
        setConfirmNewPassword('');
      } else {
        const data = await response.json();
        alert(`Failed to update password: ${data.error || response.statusText}`);
      }
    } catch (error) {
      console.error('Error updating password:', error);
      alert('An error occurred while updating the password.');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/'); // Redirect to Landing Page
  };

  const handleUnsubscribeClick = () => {
    setShowUnsubscribeModal(true);
  };

  const handleCloseModal = () => {
    setShowUnsubscribeModal(false);
  };

  const handleUnsubscribeSuccess = () => {
    setUser((prevUser) => ({
      ...prevUser,
      subscriptionStatus: 'Cancelled',
    }));
  };

  return (
    <>
      <main className="flex flex-col items-center px-4 pb-10 mx-auto w-full max-w-7xl min-h-screen pt-20">
        <div className="app-container flex flex-col items-center w-full">
          {/* Header */}
          <Header title="Account Settings" />

          {/* Subscription Status */}
          <section className="mt-6 w-full">
            <h2 className="text-2xl text-white">Subscription Status:</h2>
            <p className="mt-2 text-xl text-white">{user.subscriptionStatus}</p>
          </section>

          {/* Profile Update Form */}
          <form className="w-full mt-8" onSubmit={(e) => e.preventDefault()}>
            <h2 className="text-xl text-white mb-4">Update Email</h2>
            <div className="mt-4">
              <label className="text-white">Email:</label>
              <input
                type="email"
                className="w-full mt-2 px-4 py-2 bg-gray-200 rounded-lg text-black"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <Button
              text="Update Email"
              className="mt-6 bg-zinc-800 text-white"
              onClick={handleUpdateProfile}
            />
          </form>

          {/* Password Update Form */}
          <form className="w-full mt-8" onSubmit={(e) => e.preventDefault()}>
            <h2 className="text-xl text-white mb-4">Change Password</h2>
            <div className="mt-4">
              <label className="text-white">Current Password:</label>
              <input
                type="password"
                className="w-full mt-2 px-4 py-2 bg-gray-200 rounded-lg text-black"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
              />
            </div>

            <div className="mt-4">
              <label className="text-white">New Password:</label>
              <input
                type="password"
                className="w-full mt-2 px-4 py-2 bg-gray-200 rounded-lg text-black"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
              />
            </div>

            <div className="mt-4">
              <label className="text-white">Confirm New Password:</label>
              <input
                type="password"
                className="w-full mt-2 px-4 py-2 bg-gray-200 rounded-lg text-black"
                value={confirmNewPassword}
                onChange={(e) => setConfirmNewPassword(e.target.value)}
              />
            </div>

            <Button
              text="Update Password"
              className="mt-6 bg-zinc-800 text-white"
              onClick={handleUpdatePassword}
            />
          </form>

          {/* Unsubscribe and Logout Buttons */}
          <Button
            text="Unsubscribe"
            className="mt-4 bg-red-600 text-white"
            onClick={handleUnsubscribeClick}
          />

          <Button
            text="Log Out"
            className="mt-4 bg-neutral-200 text-black"
            onClick={handleLogout}
          />

          {/* Unsubscribe Modal */}
          {showUnsubscribeModal && (
            <UnsubscribeModal
              isOpen={showUnsubscribeModal}
              onClose={handleCloseModal}
              userEmail={user.email}
              onUnsubscribeSuccess={handleUnsubscribeSuccess}
            />
          )}
        </div>
      </main>
      <BottomNav />
    </>
  );
};

export default AccountSettingsPage;
