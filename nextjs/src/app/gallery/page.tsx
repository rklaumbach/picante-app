// src/app/gallery/page.tsx

'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useSession } from 'next-auth/react';
import Header from '../../components/Header';
import DeleteConfirmationDialog from '../../components/DeleteConfirmationDialog';
import BottomNav from '../../components/BottomNav';
import ImageModal from '../../components/ImageModal';
import { useRouter } from 'next/navigation';
import Button from '../../components/Button';
import { toast } from 'react-toastify';
import { ImageData } from '@/types/types';
import Image from 'next/image';
import CachedImage from '@/components/CachedImage';
import { useInView } from 'react-intersection-observer';
import { FixedSizeGrid } from 'react-window';
import { debounce } from 'lodash'; 


const GalleryPage: React.FC = () => {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [images, setImages] = useState<ImageData[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [hasMore, setHasMore] = useState<boolean>(true);
  const [nextCursor, setNextCursor] = useState<string | null>(null);

  const [gridConfig, setGridConfig] = useState({
    columnCount: 4,
    columnWidth: 200,
    rowHeight: 200,
    containerWidth: 400,
    containerHeight: 600,
  });

  const { ref, inView } = useInView({
    threshold: 0,
  });

  const fetchImages = useCallback(
    debounce(async () => {
      if (status !== 'authenticated' || loading || !hasMore) return;
  
      setLoading(true);
      try {
        const url = new URL('/api/images/gallery', window.location.origin);
        url.searchParams.append('limit', '20');
        if (nextCursor) {
          url.searchParams.append('cursor', nextCursor);
        }
  
        const response = await fetch(url.toString(), {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
        });
  
        if (response.ok) {
          const data = await response.json();
          if (Array.isArray(data.images)) {
            setImages((prev) => [...prev, ...data.images]);
            setNextCursor(data.nextCursor || null);
            setHasMore(Boolean(data.nextCursor));
          } else {
            console.error('Invalid data format for images:', data.images);
            setHasMore(false);
          }
        } else {
          console.error('Failed to fetch images:', response.status, response.statusText);
          setHasMore(false);
        }
      } catch (error) {
        console.error('Error fetching images:', error);
        setHasMore(false);
      } finally {
        setLoading(false);
      }
    }, 300),
    [status, nextCursor, loading, hasMore]
  );
  

  useEffect(() => {
    if (status === 'authenticated') {
      fetchImages();
    } else if (status === 'unauthenticated') {
      router.push('/'); // Redirect unauthenticated users
    }
  }, [status, fetchImages, router]);

  useEffect(() => {
    if (inView && hasMore && !loading) {
      fetchImages();
    }
  }, [inView, hasMore, loading, fetchImages]);

  useEffect(() => {
    const updateGridConfig = () => {
      const width = window.innerWidth;
  
      if (width < 640) {
        setGridConfig({
          columnCount: 1,
          columnWidth: width - 16,
          rowHeight: 200,
          containerWidth: width,
          containerHeight: 400,
        });
      } else if (width < 1024) {
        setGridConfig({
          columnCount: 2,
          columnWidth: (width - 32) / 2,
          rowHeight: 250,
          containerWidth: width,
          containerHeight: 500,
        });
      } else {
        setGridConfig({
          columnCount: 4,
          columnWidth: (width - 64) / 4,
          rowHeight: 300,
          containerWidth: width,
          containerHeight: 600,
        });
      }
    };
  
    updateGridConfig();
    window.addEventListener('resize', updateGridConfig);
    return () => window.removeEventListener('resize', updateGridConfig);
  }, []);
  
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

  if (status === 'loading' || (status === 'authenticated' && images.length === 0 && loading)) {
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
          <div className="w-full h-full mt-4 overflow-y-auto overflow-x-hidden px-2">
            {images.length === 0 ? (
              <p className="text-white">No images found. Start generating and saving your images!</p>
            ) : (
              <FixedSizeGrid
              columnCount={4} // Adjust based on your desired number of columns
              rowCount={Math.ceil(images.length / 4)} // Calculate the rows
              columnWidth={200} // Adjust based on your design
              rowHeight={200} // Set height of each row
              height={600} // Container height
              width={800} // Container width
              >
                {({ columnIndex, rowIndex, style }) => {
                  const imageIndex = rowIndex * 4 + columnIndex;
                  if (imageIndex >= images.length) return null;

                  const image = images[imageIndex];
                  return (
                    <div style={style}>
                      <img
                        className="rounded-lg transform transition-transform duration-200 hover:scale-105"
                        src={image.image_url}
                        alt={image.filename}
                      />
                    </div>
                  );
                }}
                </FixedSizeGrid>

            )}
            {/* Sentinel for Infinite Scroll */}
            {hasMore && (
              <div ref={ref} className="h-10">
                {loading && <p className="text-center text-gray-500">Loading more images...</p>}
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