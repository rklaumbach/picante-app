// src/app/account/page.tsx

'use client';

import React, { useState, useEffect } from 'react';
import { useSession, signIn, signOut } from 'next-auth/react';
import Header from '../../components/Header';
import Button from '../../components/Button';
import UnsubscribeModal from '../../components/UnsubscribeModal';
import BottomNav from '../../components/BottomNav';
import { useRouter } from 'next/navigation';

interface User {
  id: string;
  email: string;
  credits: number;
  subscriptionStatus: string;
}

const AccountSettingsPage: React.FC = () => {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [email, setEmail] = useState<string>('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [showUnsubscribeModal, setShowUnsubscribeModal] = useState(false);

  useEffect(() => {
    const fetchUser = async () => {
      if (status === 'authenticated') {
        try {
          const response = await fetch('/api/user/profile', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
          });

          if (response.ok) {
            const data = await response.json();
            setUser(data.user);
            setEmail(data.user.email);
          } else {
            console.error('Failed to fetch user data');
            router.push('/');
          }
        } catch (error) {
          console.error('Error fetching user data:', error);
          router.push('/');
        } finally {
          setLoading(false);
        }
      } else if (status === 'unauthenticated') {
        router.push('/');
      }
    };

    fetchUser();
  }, [status, router]);

  // Function to update user profile (email)
  const handleUpdateProfile = async () => {
    try {
      const response = await fetch('/api/auth/update-email', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ email }),
      });

      if (response.ok) {
        const data = await response.json();
        alert(data.message);
        // Optionally, refresh user data
        const userResponse = await fetch('/api/user/profile', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
        });

        if (userResponse.ok) {
          const updatedData = await userResponse.json();
          setUser(updatedData.user);
        }
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
    if (newPassword !== confirmNewPassword) {
      alert('New passwords do not match.');
      return;
    }

    try {
      const response = await fetch('/api/auth/update-password', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ currentPassword, newPassword }),
      });

      if (response.ok) {
        const data = await response.json();
        alert(data.message);
        // Optionally, sign out the user to require re-authentication
        await signOut({ callbackUrl: '/' });
      } else {
        const data = await response.json();
        alert(`Failed to update password: ${data.error || response.statusText}`);
      }
    } catch (error) {
      console.error('Error updating password:', error);
      alert('An error occurred while updating the password.');
    }
  };

  // Function to handle unsubscribe success
  const handleUnsubscribeSuccess = () => {
    if (user) {
      setUser({ ...user, subscriptionStatus: 'inactive' });
    }
    alert('You have successfully unsubscribed.');
  };

  // Loading state
  if (loading) {
    return (
      <>
        <main className="flex flex-col items-center px-4 pb-10 mx-auto w-full max-w-7xl min-h-screen pt-20">
          <div className="app-container flex flex-col items-center w-full">
            <Header title="Account Settings" />
            <p className="mt-6 text-white">Loading...</p>
          </div>
        </main>
        <BottomNav />
      </>
    );
  }

  return (
    <>
      <main className="flex flex-col items-center px-4 pb-10 mx-auto w-full max-w-7xl min-h-screen pt-20">
        <div className="app-container flex flex-col items-center w-full">
          {/* Header */}
          <Header title="Account Settings" />

          {/* Subscription Status */}
          <section className="mt-6 w-full">
            <h2 className="text-2xl text-white">Subscription Status:</h2>
            <p className="mt-2 text-xl text-white">{user?.subscriptionStatus}</p>
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
                required
              />
            </div>

            <Button
              text="Update Email"
              className="mt-6 bg-zinc-800 text-white"
              onClick={handleUpdateProfile}
              disabled={!email.trim()}
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
                required
              />
            </div>

            <div className="mt-4">
              <label className="text-white">New Password:</label>
              <input
                type="password"
                className="w-full mt-2 px-4 py-2 bg-gray-200 rounded-lg text-black"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
            </div>

            <div className="mt-4">
              <label className="text-white">Confirm New Password:</label>
              <input
                type="password"
                className="w-full mt-2 px-4 py-2 bg-gray-200 rounded-lg text-black"
                value={confirmNewPassword}
                onChange={(e) => setConfirmNewPassword(e.target.value)}
                required
              />
            </div>

            <Button
              text="Update Password"
              className="mt-6 bg-zinc-800 text-white"
              onClick={handleUpdatePassword}
              disabled={
                !currentPassword.trim() ||
                !newPassword.trim() ||
                !confirmNewPassword.trim()
              }
            />
          </form>

          {/* Unsubscribe and Logout Buttons */}
          <Button
            text="Unsubscribe"
            className="mt-4 bg-red-600 text-white"
            onClick={() => setShowUnsubscribeModal(true)}
          />

          <Button
            text="Log Out"
            className="mt-4 bg-neutral-200 text-black"
            onClick={() => signOut({ callbackUrl: '/' })}
          />

          {/* Unsubscribe Modal */}
          {showUnsubscribeModal && (
            <UnsubscribeModal
              isOpen={showUnsubscribeModal}
              onClose={() => setShowUnsubscribeModal(false)}
              userEmail={user?.email || ''}
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
