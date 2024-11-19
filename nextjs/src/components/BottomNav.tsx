// src/components/BottomNav.tsx

'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useSession, signIn, signOut } from 'next-auth/react';
import Button from './Button';
import LoginDialog from './LoginDialog'; // Ensure this component exists
import SignUpDialog from './SignUpDialog'; // Ensure this component exists

const BottomNav: React.FC = () => {
  const { data: session, status } = useSession();
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [isSignUpOpen, setIsSignUpOpen] = useState(false);

  const openLogin = () => setIsLoginOpen(true);
  const closeLogin = () => setIsLoginOpen(false);

  const openSignUp = () => setIsSignUpOpen(true);
  const closeSignUp = () => setIsSignUpOpen(false);

  const handleSignOut = () => {
    signOut({ callbackUrl: '/' });
  };

  return (
    <>
      <nav className="fixed bottom-0 left-0 right-0 bg-gray-800 p-4 flex justify-around">
        <Link href="/generate" className="text-white">
          Generate
        </Link>
        <Link href="/gallery" className="text-white">
          Gallery
        </Link>
        <Link href="/account" className="text-white">
          Account
        </Link>
        {session ? (
          <Button text="Sign Out" className="bg-red-500 text-white px-4 py-2" onClick={handleSignOut} />
        ) : (
          <>
            <Button text="Sign In" className="bg-blue-500 text-white px-4 py-2" onClick={openLogin} />
            <Button text="Sign Up" className="bg-green-500 text-white px-4 py-2 ml-2" onClick={openSignUp} />
          </>
        )}
      </nav>

      {/* Login Dialog */}
      <LoginDialog isOpen={isLoginOpen} onClose={closeLogin} />

      {/* Sign-Up Dialog */}
      <SignUpDialog isOpen={isSignUpOpen} onClose={closeSignUp} />
    </>
  );
};

export default BottomNav;
