// frontend/src/pages/LandingPage.tsx

'use client';
import React, { useState } from 'react';
import Header from '../components/Header';
import Button from '../components/Button';
import FeatureList from '../components/FeatureList';
import LoginDialog from '../components/LoginDialog';
import SignUpDialog from '../components/SignUpDialog';

const LandingPage: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [isSignUpOpen, setIsSignUpOpen] = useState(false);

  return (
    <>
      <Header title="Welcome to Picante" />
      <main className="flex flex-col items-center px-4 pb-10 mx-auto w-full max-w-7xl min-h-screen pt-20">
        <div className="app-container flex flex-col items-center w-full">
          {/* Welcome Message */}
          <div className="flex flex-col items-center text-center">
            <h2 className="text-2xl sm:text-3xl md:text-4xl font-semibold mt-4 text-white">
              Generate stunning images with AI.
            </h2>
            <p className="mt-2 text-base sm:text-lg md:text-xl max-w-4xl text-white">
              Subscribe to get unlimited access and exclusive features.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="mt-6 w-full max-w-md sm:max-w-lg flex flex-col space-y-4">
            <Button
              text="Get Started"
              className="bg-blue-500 text-white w-full py-3 rounded-lg hover:bg-blue-600 transition duration-200"
              onClick={() => setIsSignUpOpen(true)} // Open Sign Up dialog
            />
            <Button
              text="Log In"
              className="bg-neutral-200 text-black w-full py-3 rounded-lg hover:bg-neutral-300 transition duration-200"
              onClick={() => setIsLoginOpen(true)}
            />
          </div>

          {/* Feature List */}
          <FeatureList
            features={[
              'Unlimited Images',
              'Priority Processing',
              'AI Upscaling',
              'Access to Animation Features',
            ]}
          />
        </div>
      </main>

      {/* Dialogs */}
      <LoginDialog
        isOpen={isLoginOpen}
        onClose={() => setIsLoginOpen(false)}
      />
      <SignUpDialog
        isOpen={isSignUpOpen}
        onClose={() => setIsSignUpOpen(false)}
      />
    </>
  );
};

export default LandingPage;
