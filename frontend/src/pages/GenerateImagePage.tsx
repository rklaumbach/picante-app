// frontend/src/pages/GenerateImagePage.tsx

import React, { useState } from 'react';
import Header from '../components/Header';
import Button from '../components/Button';
import PromptInput from '../components/PromptInput';
import ImageDisplay from '../components/ImageDisplay';
import OutOfCreditsDialog from '../components/OutOfCreditsDialog';
import LoginDialog from '../components/LoginDialog';
import BottomNav from '../components/BottomNav';
import { useAuth } from '../contexts/AuthContext';

const GenerateImagePage: React.FC = () => {
  const [bodyPrompt, setBodyPrompt] = useState('');
  const [facePrompt, setFacePrompt] = useState('');
  const [generatedImage, setGeneratedImage] = useState('');
  const [isOutOfCreditsDialogOpen, setIsOutOfCreditsDialogOpen] = useState(false);
  const [isLoginDialogOpen, setIsLoginDialogOpen] = useState(false);
  const { isAuthenticated } = useAuth();

  const handleGenerateImage = async () => {
    if (!isAuthenticated) {
      setIsLoginDialogOpen(true);
      return;
    }

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setIsLoginDialogOpen(true);
        return;
      }

      const response = await fetch('/api/images/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ bodyPrompt, facePrompt }),
      });

      const data = await response.json();

      if (response.ok) {
        setGeneratedImage(data.imageUrl);
      } else if (response.status === 403 && data.error === 'Out of credits') {
        setIsOutOfCreditsDialogOpen(true);
      } else {
        alert(`Failed to generate image: ${data.error || response.statusText}`);
      }
    } catch (error) {
      console.error('Error generating image:', error);
      alert('An error occurred while generating the image.');
    }
  };

  return (
    <>
      <main className="flex flex-col items-center px-4 pb-16 mx-auto w-full max-w-7xl min-h-screen pt-20">
        <div className="app-container flex flex-col items-center w-full">
          {/* Header */}
          <Header title="Generate Image" />

          {/* Prompt Inputs */}
          <div className="w-full mt-6">
            <PromptInput
              label="Body Prompt"
              value={bodyPrompt}
              onChange={setBodyPrompt}
            />
            <PromptInput
              label="Face Prompt"
              value={facePrompt}
              onChange={setFacePrompt}
            />
          </div>

          {/* Generate Button */}
          <Button
            text="Generate"
            className="mt-6 bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 transition duration-200"
            onClick={handleGenerateImage}
          />

          {/* Display Generated Image */}
          {generatedImage && <ImageDisplay imageUrl={generatedImage} />}
        </div>
      </main>
      <OutOfCreditsDialog
        isOpen={isOutOfCreditsDialogOpen}
        onClose={() => setIsOutOfCreditsDialogOpen(false)}
      />
      <LoginDialog
        isOpen={isLoginDialogOpen}
        onClose={() => setIsLoginDialogOpen(false)}
      />
      <BottomNav />
    </>
  );
};

export default GenerateImagePage;
