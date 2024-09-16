// frontend/src/pages/AccountSettingsPage.tsx

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
  const [showUnsubscribeModal, setShowUnsubscribeModal] = useState(false);

  useEffect(() => {
    // Fetch user data
    const fetchUserData = async () => {
      try {
        const response = await fetch('/api/user/profile', {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
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
  }, [navigate]);

  const handleUpdateAccount = async () => {
    // Implement account update logic
    alert('Account information updated successfully!');
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

          {/* Account Update Form */}
          <form className="w-full mt-8" onSubmit={(e) => e.preventDefault()}>
            <div className="mt-4">
              <label className="text-white">Email:</label>
              <input
                type="email"
                className="w-full mt-2 px-4 py-2 bg-gray-200 rounded-lg text-black"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

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

            <Button
              text="Update Account"
              className="mt-6 bg-zinc-800 text-white"
              onClick={handleUpdateAccount}
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
