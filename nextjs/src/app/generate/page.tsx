// src/app/generate/page.tsx

'use client';

import React, { useState, useEffect } from 'react';
import { useSession, signIn, signOut } from 'next-auth/react';
import Header from '../../components/Header';
import Button from '../../components/Button';
import ImageDisplay from '../../components/ImageDisplay';
import ImageModal from '../../components/ImageModal';
import OutOfCreditsDialog from '../../components/OutOfCreditsDialog';
import BottomNav from '../../components/BottomNav';
import TagAutocomplete from '../../components/TagAutocomplete';
import { useRouter } from 'next/navigation';
import { ImageData } from '@/types/types'; // Import the unified Image interface


// Define the structure for each selected tag with its weight
interface SelectedTag {
  tag: string;
  weight: number;
}

interface UserCredits {
  credits: number;
}

const GenerateImagePage: React.FC = () => {
  const { data: session, status } = useSession();
  const router = useRouter();

  const [bodyTags, setBodyTags] = useState<SelectedTag[]>([]);
  const [faceTags, setFaceTags] = useState<SelectedTag[]>([]);
  const [generatedImage, setGeneratedImage] = useState('');
  const [isOutOfCreditsDialogOpen, setIsOutOfCreditsDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [credits, setCredits] = useState<number>(0); // Initialize to 0
  const [width, setWidth] = useState<number>(1024); // Default resolution
  const [height, setHeight] = useState<number>(1024); // Default resolution

  const [upscaleEnabled, setUpscaleEnabled] = useState<boolean>(false);
  const [upscaleFactor, setUpscaleFactor] = useState<2 | 4>(2);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [isImageModalOpen, setIsImageModalOpen] = useState<boolean>(false);


  // Fetch user credits upon session availability
  useEffect(() => {
    const fetchCredits = async () => {
      if (session?.user?.id) {
        try {
          const response = await fetch('/api/user/credits', {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
              // Assuming authentication is handled via cookies or tokens
            },
          });

          if (response.ok) {
            const data: UserCredits = await response.json();
            setCredits(data.credits);
          } else {
            console.error('Failed to fetch user credits');
            setCredits(0);
          }
        } catch (error) {
          console.error('Error fetching user credits:', error);
          setCredits(0);
        }
      }
    };

    if (status === 'authenticated') {
      fetchCredits();
    }
  }, [session, status]);

  const handleGenerateImage = async () => {
    console.log('handleGenerateImage triggered'); // Debugging line

    if (bodyTags.length === 0 || faceTags.length === 0) {
      alert('Please provide both Body Prompt and Face Prompt.');
      return;
    }

    if (credits <= 0) {
      setIsOutOfCreditsDialogOpen(true);
      return;
    }

    setIsLoading(true);
    setGeneratedImage(''); // Clear previous image

    try {
      // Prepare prompts by joining tags with commas and weights in "<tag>:<weight>" format
      const bodyPrompt = bodyTags.map((t) => `(${t.tag}:${t.weight})`).join(', ');
      const facePrompt = faceTags.map((t) => `(${t.tag}:${t.weight})`).join(', ');

      // Extract user_id from session
      const userId = session?.user?.id;
      if (!userId) {
        alert('User ID not found.');
        setIsLoading(false);
        return;
      }

      // Submit the job to the Next.js API route
      const response = await fetch('/api/images/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Authentication headers if required
        },
        body: JSON.stringify({ 
          user_id: userId, 
          bodyPrompt, 
          facePrompt, 
          width,
          height, 
          upscaleEnabled, 
          upscaleFactor 
        }),
      });

      const data = await response.json();

      console.log('Response from /api/images/generate:', data); // Debugging line

      if (response.ok) {
        const { job_id, status } = data;
        if (status === 'completed') {
          // Assuming backend sends image_urls in the response when completed
          if (data.image_urls && data.image_urls.final_image) {
            setGeneratedImage(data.image_urls.final_image);
            setCredits((prev) => prev - 1);
          } else {
            alert('Image generation completed but image URL not found.');
          }
          setIsLoading(false);
        } else if (status === 'queued') {
          // Poll for job status
          const interval = setInterval(async () => {
            try {
              const statusResponse = await fetch(`/api/images/job-status?job_id=${job_id}`, {
                method: 'GET',
                headers: {
                  'Content-Type': 'application/json',
                  // Authentication headers if required
                },
              });

              const statusData = await statusResponse.json();

              console.log('Polling job status:', statusData); // Debugging line

              if (statusResponse.ok) {
                if (statusData.status === 'completed') {
                  clearInterval(interval);
                  if (statusData.image_urls && statusData.image_urls.final_image) {
                    setGeneratedImage(statusData.image_urls.final_image);
                    setCredits((prev) => prev - 1);
                  } else {
                    alert('Final image URL not found.');
                  }
                  setIsLoading(false);
                } else if (statusData.status === 'failed') {
                  clearInterval(interval);
                  alert(`Image generation failed: ${statusData.reason || 'Unknown error.'}`);
                  setIsLoading(false);
                }
                // If status is still 'queued', continue polling
              } else {
                clearInterval(interval);
                alert(`Failed to fetch job status: ${statusData.error || 'Unknown error.'}`);
                setIsLoading(false);
              }
            } catch (error) {
              console.error('Error polling job status:', error);
              clearInterval(interval);
              alert('An error occurred while checking job status.');
              setIsLoading(false);
            }
          }, 3000); // Poll every 3 seconds
        } else {
          alert('Unexpected job status.');
          setIsLoading(false);
        }
      } else if (response.status === 403 && data.error === 'Out of credits') {
        setIsOutOfCreditsDialogOpen(true);
        setIsLoading(false);
      } else {
        alert(`Failed to generate image: ${data.error || response.statusText}`);
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Error generating image:', error);
      alert('An error occurred while generating the image.');
      setIsLoading(false);
    }
  };

  const handleGeneratedImageClick = () => {
    const image: ImageData = {
      id: '', // No ID since it's not saved yet
      image_url: generatedImage,
      filename: 'Generated Image',
      body_prompt: bodyTags.map((t) => `${t.tag}:${t.weight}`).join(', '),
      face_prompt: faceTags.map((t) => `${t.tag}:${t.weight}`).join(', '),
      width: width,
      height: height,
      created_at: new Date().toISOString(),
    };
    setSelectedImage(image);
    setIsImageModalOpen(true);
  };

  const handleDownload = async (image: ImageData) => {
    try {
      const response = await fetch(image.image_url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
  
      const link = document.createElement('a');
      link.href = url;
      link.download = image.filename || 'generated_image.png';
      document.body.appendChild(link);
      link.click();
  
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading the image:', error);
      // Handle error (e.g., show a toast notification)
    }
  };
  
  const handleFullView = (image: ImageData) => {
    window.open(image.image_url, '_blank');
  };

  // Handle sign-in
  const handleSignIn = () => {
    signIn(); // Redirects to NextAuth.js sign-in page
  };

  // Handle sign-out
  const handleSignOut = () => {
    signOut({ callbackUrl: '/' }); // Redirects to landing page after sign-out
  };

  // Redirect unauthenticated users to sign-in
  useEffect(() => {
    if (status === 'loading') return; // Do nothing while loading
    if (status === 'unauthenticated') {
      handleSignIn();
    }
  }, [status]);

  // If still loading authentication status
  if (status === 'loading') {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-300">
        <p className="text-xl">Loading...</p>
      </div>
    );
  }

  return (
    <>
      <main className="flex flex-col items-center px-8 pb-20 mx-auto w-full max-w-7xl min-h-screen pt-24">
        <div className="app-container flex flex-col items-center w-full">
          {/* Header */}
          <Header title="Generate Image" />

          {/* Sign Out Button */}
          <div className="fixed top-6 left-6">
            <Button
              text="Sign Out"
              className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition duration-200"
              onClick={handleSignOut}
            />
          </div>

          {/* Credits Display */}
          <div className="fixed top-6 right-6 bg-gray-200 p-4 rounded shadow-lg text-xl font-semibold">
            Credits: {credits}
          </div>

          {/* Prompt Inputs */}
          <div className="w-full mt-8">
            <TagAutocomplete
              label="Body Prompt"
              selectedTags={bodyTags}
              setSelectedTags={setBodyTags}
            />
            <TagAutocomplete
              label="Face Prompt"
              selectedTags={faceTags}
              setSelectedTags={setFaceTags}
            />
          </div>

          {/* Resolution Selection */}
          <div className="w-full mt-6">
            <label className="block text-2xl font-semibold text-gray-700 mb-3">Select Resolution</label>
            <div className="flex flex-col sm:flex-row sm:space-x-8 space-y-4 sm:space-y-0">
              <label className="inline-flex items-center text-xl">
                <input
                  type="radio"
                  name="resolution"
                  value="720x1280"
                  checked={width === 720 && height === 1280}
                  onChange={(e) => {
                    const [w, h] = e.target.value.split('x').map(Number);
                    setWidth(w);
                    setHeight(h);
                  }}
                  className="form-radio h-6 w-6 text-blue-600"
                />
                <span className="ml-4">Portrait (9:16)</span>
              </label>
              <label className="inline-flex items-center text-xl">
                <input
                  type="radio"
                  name="resolution"
                  value="1024x1024"
                  checked={width === 1024 && height === 1024}
                  onChange={(e) => {
                    const [w, h] = e.target.value.split('x').map(Number);
                    setWidth(w);
                    setHeight(h);
                  }}
                  className="form-radio h-6 w-6 text-blue-600"
                />
                <span className="ml-4">Square (1:1)</span>
              </label>
              <label className="inline-flex items-center text-xl">
                <input
                  type="radio"
                  name="resolution"
                  value="1280x720"
                  checked={width === 1280 && height === 720}
                  onChange={(e) => {
                    const [w, h] = e.target.value.split('x').map(Number);
                    setWidth(w);
                    setHeight(h);
                  }}
                  className="form-radio h-6 w-6 text-blue-600"
                />
                <span className="ml-4">Landscape (16:9)</span>
              </label>
              {/* Add more resolutions as needed */}
            </div>
          </div>

          {/* Upscale Options */}
          <div className="w-full mt-6">
            <label className="block text-2xl font-semibold text-gray-700 mb-3">Upscale Options</label>
            <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-8 space-y-4 sm:space-y-0">
              <label className="inline-flex items-center text-xl">
                <input
                  type="checkbox"
                  checked={upscaleEnabled}
                  onChange={(e) => setUpscaleEnabled(e.target.checked)}
                  className="form-checkbox h-6 w-6 text-blue-600"
                />
                <span className="ml-4">Enable Upscaling</span>
              </label>
              {upscaleEnabled && (
                <div className="flex flex-col sm:flex-row sm:space-x-8 space-y-4 sm:space-y-0">
                  <label className="inline-flex items-center text-xl">
                    <input
                      type="radio"
                      name="upscaleFactor"
                      value="2"
                      checked={upscaleFactor === 2}
                      onChange={(e) => setUpscaleFactor(parseInt(e.target.value) as 2 | 4)}
                      className="form-radio h-6 w-6 text-blue-600"
                    />
                    <span className="ml-4">2x</span>
                  </label>
                  <label className="inline-flex items-center text-xl">
                    <input
                      type="radio"
                      name="upscaleFactor"
                      value="4"
                      checked={upscaleFactor === 4}
                      onChange={(e) => setUpscaleFactor(parseInt(e.target.value) as 2 | 4)}
                      className="form-radio h-6 w-6 text-blue-600"
                    />
                    <span className="ml-4">4x</span>
                  </label>
                </div>
              )}
            </div>
          </div>

          {/* Generate Button */}
          <Button
            text={isLoading ? 'Generating...' : 'Generate'}
            className={`mt-8 px-8 py-6 rounded-lg text-2xl font-semibold transition duration-200 ${
              isLoading
                ? 'bg-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
            onClick={handleGenerateImage}
            disabled={isLoading}
          />

          {/* Display Generated Image */}
          {generatedImage && (
            <div onClick={handleGeneratedImageClick}>
              <ImageDisplay imageUrl={generatedImage} width={width} height={height} />
            </div>
          )}

          {isImageModalOpen && selectedImage && (
            <ImageModal
              image={selectedImage}
              onClose={() => setIsImageModalOpen(false)}
              onDownload={() => handleDownload(selectedImage)}
              onFullView={() => handleFullView(selectedImage)}
              // Omit the onDelete prop
            />
          )}
        </div>
      </main>
      <OutOfCreditsDialog
        isOpen={isOutOfCreditsDialogOpen}
        onClose={() => setIsOutOfCreditsDialogOpen(false)}
      />
      <BottomNav />
    </>
  );
};

export default GenerateImagePage;
