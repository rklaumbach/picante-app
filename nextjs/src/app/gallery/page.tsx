// src/app/gallery/page.tsx

'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useSession } from 'next-auth/react';
import Header from '../../components/Header';
import DeleteConfirmationDialog from '../../components/DeleteConfirmationDialog';
import BottomNav from '../../components/BottomNav';
import ImageModal from '../../components/ImageModal';
import { useRouter } from 'next/navigation';
import Button from '../../components/Button';
import { toast } from 'react-toastify'; // Import toast for notifications
import { Image } from '@/types/types'; // Import the unified Image interface

const GalleryPage: React.FC = () => {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [images, setImages] = useState<Image[]>([]);
  const [selectedImage, setSelectedImage] = useState<Image | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchImages = async () => {
      if (status === 'authenticated') {
        try {
          const response = await fetch('/api/images/gallery', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
          });

          if (response.ok) {
            const data = await response.json();
            if (Array.isArray(data.images)) {
              setImages(data.images);
            } else {
              console.error('Invalid data format for images:', data.images);
            }
          } else {
            console.error('Failed to fetch images:', response.status, response.statusText);
          }
        } catch (error) {
          console.error('Error fetching images:', error);
        } finally {
          setLoading(false);
        }
      } else if (status === 'unauthenticated') {
        router.push('/'); // Redirect unauthenticated users
      }
    };

    fetchImages();
  }, [status, router]);

  const handleImageClick = (image: Image) => {
    setSelectedImage(image);
  };

  const handleDeleteClick = (imageId: string) => {
    setImageToDelete(imageId);
    setIsDeleteDialogOpen(true);
  };

  const handleConfirmDelete = useCallback(async () => {
    if (!imageToDelete) return;

    try {
      const response = await fetch(`/api/images/${imageToDelete}`, {
        method: 'DELETE',
        credentials: 'include',
      });

      if (response.ok) {
        setImages(images.filter((img) => img.id !== imageToDelete));
        setIsDeleteDialogOpen(false);
        setImageToDelete(null);
        toast.success('Image deleted successfully!');
      } else {
        const data = await response.json();
        toast.error(`Failed to delete image: ${data.error || response.statusText}`);
      }
    } catch (error) {
      console.error('Error deleting image:', error);
      toast.error('An error occurred while deleting the image.');
    }
  }, [imageToDelete, images]);

  // Function to handle image download
  const handleDownload = useCallback((image: Image) => {
    const link = document.createElement('a');
    link.href = image.image_url;
    link.download = image.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // Function to handle full view
  const handleFullView = useCallback((image: Image) => {
    // Open the full view in a new tab and pass the image data via sessionStorage
    window.open(image.image_url, '_blank');
  }, []);

  if (loading) {
    return (
      <>
        <main className="flex flex-col items-center px-7 pb-16 mx-auto w-full max-w-7xl min-h-screen pt-20">
          <div className="app-container flex flex-col items-center w-full">
            <Header title="Gallery" />
            <p className="mt-6 text-white">Loading...</p>
          </div>
        </main>
        <BottomNav />
      </>
    );
  }

  return (
    <>
      <main className="flex flex-col items-center px-7 pb-16 mx-auto w-full max-w-7xl min-h-screen pt-20">
        <div className="app-container flex flex-col items-center w-full">
          {/* Header */}
          <Header title="Gallery" />

          {/* Smaller Title Above the Gallery */}
          <h2 className="text-2xl font-semibold mt-6 text-black">Your Images:</h2>

          {/* Scrollable Image Gallery */}
          <div className="w-full mt-4 max-h-[70vh] overflow-y-auto overflow-x-hidden px-2">
            {images.length === 0 ? (
              <p className="text-white">No images found. Start generating and saving your images!</p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                {images.map((image) => (
                  <div
                    key={image.id}
                    className="relative group cursor-pointer overflow-hidden"
                    onClick={() => handleImageClick(image)}
                  >
                    <img
                      src={image.image_url}
                      alt={image.filename}
                      loading="lazy"
                      className="w-full h-auto rounded-lg object-cover transform transition-transform duration-200 hover:scale-105"
                    />
                    <button
                      className="absolute top-2 right-2 bg-red-600 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent triggering the image click
                        handleDeleteClick(image.id);
                      }}
                      aria-label="Delete Image"
                    >
                      ✖
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Image Overlay Modal */}
          {selectedImage && (
            <ImageModal
              image={selectedImage}
              onClose={() => setSelectedImage(null)}
              onDownload={() => handleDownload(selectedImage)}
              onDelete={() => handleDeleteClick(selectedImage.id)}
              onFullView={() => handleFullView(selectedImage)}
            />
          )}

          {/* Delete Confirmation Dialog */}
          <DeleteConfirmationDialog
            isOpen={isDeleteDialogOpen}
            onClose={() => setIsDeleteDialogOpen(false)}
            onConfirm={handleConfirmDelete}
          />
        </div>
      </main>
      <BottomNav />
    </>
  );
};

export default GalleryPage;
