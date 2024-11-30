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
import { FixedSizeGrid as Grid } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import CachedImage from '@/components/CachedImage'; // Import the CachedImage component
import { useInView } from 'react-intersection-observer';


const COLUMN_WIDTH = 250; // Adjust based on your design
const ROW_HEIGHT = 250; // Adjust based on your design

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

  const { ref, inView } = useInView({
    threshold: 0,
  });

  const fetchImages = useCallback(async () => {
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
  }, [status, nextCursor, loading, hasMore]);

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

  const Cell = ({ columnIndex, rowIndex, style }: any) => {
    const index = rowIndex * numColumns + columnIndex;
    if (index >= images.length) return null;
    const image = images[index];

    return (
      <div
        style={style}
        className="relative group cursor-pointer overflow-hidden"
        onClick={() => handleImageClick(image)}
      >
        <CachedImage imageData={image} />
        <button
          className="absolute top-2 right-2 bg-red-600 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          onClick={(e) => {
            e.stopPropagation();
            handleDeleteClick(image.id);
          }}
          aria-label="Delete Image"
        >
          ✖
        </button>
      </div>
    );
  };


  return (
    <>
      <main className="flex flex-col items-center px-7 pb-16 mx-auto w-full max-w-7xl min-h-screen pt-20">
        <div className="app-container flex flex-col items-center w-full">
          {/* Header */}
          <Header title="Gallery" />

          {/* Smaller Title Above the Gallery */}
          <h2 className="text-2xl font-semibold mt-6 text-black">Your Images:</h2>

          {/* Virtualized Image Gallery */}
          <div className="w-full h-full mt-4 px-2">
            {images.length === 0 ? (
              <p className="text-white">No images found. Start generating and saving your images!</p>
            ) : (
              <AutoSizer>
                {({ height, width }) => {
                  const numColumns = Math.floor(width / COLUMN_WIDTH);
                  const numRows = Math.ceil(images.length / numColumns);

                  return (
                    <Grid
                      columnCount={numColumns}
                      columnWidth={COLUMN_WIDTH}
                      height={height}
                      rowCount={numRows}
                      rowHeight={ROW_HEIGHT}
                      width={width}
                    >
                      {Cell}
                    </Grid>
                  );
                }}
              </AutoSizer>
            )}
          </div>

          {/* Sentinel for Infinite Scroll */}
          {hasMore && (
            <div ref={ref} className="h-10">
              {loading && <p className="text-center text-gray-500">Loading more images...</p>}
            </div>
          )}

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