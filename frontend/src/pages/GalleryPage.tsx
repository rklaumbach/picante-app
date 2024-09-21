// src/pages/GalleryPage.tsx

import React, { useState, useEffect, useCallback } from 'react';
import Header from '../components/Header';
import DeleteConfirmationDialog from '../components/DeleteConfirmationDialog';
import BottomNav from '../components/BottomNav';
import ImageModal from '../components/ImageModal';
import { useNavigate } from 'react-router-dom';

interface Image {
  _id: string;
  imageUrl: string;
  filename: string;
  bodyPrompt: string;
  facePrompt: string;
  resolution: string;
  timestamp: string;
}

// Complete mock data with nine images
const mockImages: Image[] = [
  {
    _id: '1',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+1',
    filename: 'image1.png',
    bodyPrompt: 'A serene landscape with mountains.',
    facePrompt: 'A smiling person enjoying the view.',
    resolution: '1920x1080',
    timestamp: '2024-04-25T10:30:00Z',
  },
  {
    _id: '2',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+2',
    filename: 'image2.png',
    bodyPrompt: 'A bustling city skyline at night.',
    facePrompt: 'People walking under bright lights.',
    resolution: '1920x1080',
    timestamp: '2024-04-26T12:45:00Z',
  },
  {
    _id: '3',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+3',
    filename: 'image3.png',
    bodyPrompt: 'A tranquil beach during sunset.',
    facePrompt: 'A couple holding hands by the shore.',
    resolution: '1920x1080',
    timestamp: '2024-04-27T15:20:00Z',
  },
  {
    _id: '4',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+4',
    filename: 'image4.png',
    bodyPrompt: 'A dense forest with rays of sunlight.',
    facePrompt: 'A hiker taking a break.',
    resolution: '1920x1080',
    timestamp: '2024-04-28T09:10:00Z',
  },
  {
    _id: '5',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+5',
    filename: 'image5.png',
    bodyPrompt: 'A modern art gallery interior.',
    facePrompt: 'Visitors admiring the artwork.',
    resolution: '1920x1080',
    timestamp: '2024-04-29T11:55:00Z',
  },
  {
    _id: '6',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+6',
    filename: 'image6.png',
    bodyPrompt: 'A snowy mountain peak under a clear sky.',
    facePrompt: 'A lone climber reaching the summit.',
    resolution: '1920x1080',
    timestamp: '2024-04-30T14:40:00Z',
  },
  {
    _id: '7',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+7',
    filename: 'image7.png',
    bodyPrompt: 'A vibrant marketplace with colorful stalls.',
    facePrompt: 'Shoppers exploring various goods.',
    resolution: '1920x1080',
    timestamp: '2024-05-01T16:25:00Z',
  },
  {
    _id: '8',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+8',
    filename: 'image8.png',
    bodyPrompt: 'A tranquil lake surrounded by autumn trees.',
    facePrompt: 'A fisherman casting a line.',
    resolution: '1920x1080',
    timestamp: '2024-05-02T08:15:00Z',
  },
  {
    _id: '9',
    imageUrl: 'https://via.placeholder.com/600x400.png?text=Image+9',
    filename: 'image9.png',
    bodyPrompt: 'A futuristic cityscape with flying cars.',
    facePrompt: 'Citizens navigating the high-tech environment.',
    resolution: '1920x1080',
    timestamp: '2024-05-03T13:50:00Z',
  },
];

const GalleryPage: React.FC = () => {
  const [images, setImages] = useState<Image[]>([]);
  const [selectedImageDetails, setSelectedImageDetails] = useState<Image | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState<string | null>(null);
  const navigate = useNavigate();

  // Define the BACKEND_AVAILABLE flag
  const BACKEND_AVAILABLE = true; // Set to false to use mock data

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  useEffect(() => {
    // Function to fetch images from the backend
    const fetchImages = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          console.warn('No token found, redirecting to login.');
          navigate('/login');
          return;
        }

        const response = await fetch(`${API_BASE_URL}/api/images/user`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Fetched images:', data.images); // Debugging
          if (Array.isArray(data.images)) {
            setImages(data.images);
          } else {
            console.error('Invalid data format for images:', data.images);
            // Fallback to mock data if the format is incorrect
            setImages(mockImages);
          }
        } else {
          console.error('Failed to fetch images:', response.status, response.statusText);
          // Fallback to mock data on failure
          setImages(mockImages);
        }
      } catch (error) {
        console.error('Error fetching images:', error);
        // Fallback to mock data on error
        setImages(mockImages);
      }
    };

    // Toggle between real fetch and mock data based on backend availability
    if (BACKEND_AVAILABLE) {
      fetchImages();
    } else {
      // Use mock data when backend is not available
      setImages(mockImages);
    }
  }, [navigate, BACKEND_AVAILABLE, API_BASE_URL]);

  const handleImageClick = (image: Image) => {
    setSelectedImageDetails(image);
  };

  const handleDeleteClick = (imageId: string) => {
    setImageToDelete(imageId);
    setIsDeleteDialogOpen(true);
  };

  const handleConfirmDelete = useCallback(async () => {
    if (!imageToDelete) return;

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        console.warn('No token found, redirecting to login.');
        navigate('/login');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/images/${imageToDelete}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        setImages(images.filter((img) => img._id !== imageToDelete));
        setIsDeleteDialogOpen(false);
        setImageToDelete(null);
        alert('Image deleted successfully!');
      } else {
        const data = await response.json();
        alert(`Failed to delete image: ${data.error || response.statusText}`);
      }
    } catch (error) {
      console.error('Error deleting image:', error);
      alert('An error occurred while deleting the image.');
    }
  }, [imageToDelete, images, navigate, API_BASE_URL]);

  const handleMockDelete = useCallback(() => {
    if (!imageToDelete) return;
    setImages(images.filter((img) => img._id !== imageToDelete));
    setIsDeleteDialogOpen(false);
    setImageToDelete(null);
    alert('Image deleted successfully (mock)!');
  }, [imageToDelete, images]);

  // Function to handle image download
  const handleDownload = useCallback((image: Image) => {
    const link = document.createElement('a');
    link.href = image.imageUrl;
    link.download = image.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // Function to handle full view
  const handleFullView = useCallback((image: Image) => {
    // Open the full view in a new tab and pass the image data via sessionStorage
    sessionStorage.setItem('fullViewImage', JSON.stringify(image));
    window.open('/full-view', '_blank');
  }, []);

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
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
              {images.map((image) => (
                <div
                  key={image._id}
                  className="relative group cursor-pointer overflow-hidden"
                  onClick={() => handleImageClick(image)}
                >
                  <img
                    src={image.imageUrl}
                    alt={image.filename}
                    loading="lazy"
                    className="w-full h-auto rounded-lg object-cover transform transition-transform duration-200 hover:scale-105"
                  />
                  <button
                    className="absolute top-2 right-2 bg-red-600 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent triggering the image click
                      handleDeleteClick(image._id);
                    }}
                    aria-label="Delete Image"
                  >
                    ✖
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Image Overlay Modal */}
          {selectedImageDetails && (
            <ImageModal
              image={selectedImageDetails}
              onClose={() => setSelectedImageDetails(null)}
              onDownload={handleDownload}
              onDelete={handleDeleteClick}
              onFullView={handleFullView}
            />
          )}

          {/* Delete Confirmation Dialog */}
          <DeleteConfirmationDialog
            isOpen={isDeleteDialogOpen}
            onClose={() => setIsDeleteDialogOpen(false)}
            onConfirm={BACKEND_AVAILABLE ? handleConfirmDelete : handleMockDelete}
          />
        </div>
      </main>
      <BottomNav />
    </>
  );
};

export default GalleryPage;
