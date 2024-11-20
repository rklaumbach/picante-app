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
import { ImageData } from '@/types/types'; // Import the unified Image interface
import Image from 'next/image';


const GalleryPage: React.FC = () => {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [images, setImages] = useState<ImageData[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
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

  const handleImageClick = (image: ImageData) => {
    setSelectedImage(image);
  };

  const handleDeleteClick = (imageId: string) => {
    setImageToDelete(imageId);
    setIsDeleteDialogOpen(true);
  };

  const handleConfirmDelete = useCallback(async () => {
    if (!imageToDelete) return;

    try {
      const response = await fetch(`/api/images/delete?image_id=${imageToDelete}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
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
  const handleDownload = useCallback(async (image: ImageData) => {
    try {
      // Fetch the image as a blob
      const response = await fetch(image.image_url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/octet-stream',
        },
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      // Create a temporary anchor element and trigger download
      const link = document.createElement('a');
      link.href = url;
      link.download = image.filename; // Ensure filename has the correct extension
      document.body.appendChild(link);
      link.click();

      // Clean up
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading the image:', error);
      toast.error('Failed to download the image.');
    }
  }, []);

  // Function to handle full view
  const handleFullView = useCallback((image: ImageData) => {
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
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {images.map((image) => (
                  <div
                    key={image.id}
                    className="relative group cursor-pointer overflow-hidden"
                    onClick={() => handleImageClick(image)}
                  >
                    <Image
                      src={image.image_url}
                      alt={image.filename}
                      width={image.width}
                      height={image.height}
                      layout="responsive"
                      loading="lazy"
                      className="rounded-lg transform transition-transform duration-200 hover:scale-105"
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
