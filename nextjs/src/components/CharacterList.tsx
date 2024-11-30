// src/components/CharacterList.tsx

'use client';

import React, { useEffect, useState } from 'react';
import { Character, ImageData } from '@/types/types';
import Button from './Button';
import Dialog from './Dialog';
import EnlargeCharacterImage from './EnlargeCharacterImage';
import { toast } from 'react-toastify';

interface CharacterListProps {
  onSelectCharacter: (character: Character) => void;
}

const CharacterList: React.FC<CharacterListProps> = ({ onSelectCharacter }) => {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [images, setImages] = useState<ImageData[]>([]); // State to hold user's images
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newCharacter, setNewCharacter] = useState({
    name: '',
    image_id: '', // Changed from image_path to image_id
    personality_traits: '',
    other_info: '',
  });
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const [isImageEnlarged, setIsImageEnlarged] = useState(false);

  // Fetch characters and images
  const fetchData = async () => {
    try {
      // Fetch characters
      const charactersResponse = await fetch('/api/chat/characters', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      let charactersData: Character[] = [];
      if (charactersResponse.ok) {
        const charactersJson = await charactersResponse.json();
        charactersData = charactersJson.characters;
      } else {
        console.error('Failed to fetch characters:', charactersResponse.status, charactersResponse.statusText);
        toast.error('Failed to fetch characters.');
      }

      // Fetch images
      const imagesResponse = await fetch('/api/images/user-images', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      let imagesData: ImageData[] = [];
      if (imagesResponse.ok) {
        const imagesJson = await imagesResponse.json();
        imagesData = imagesJson.images;
      } else {
        console.error('Failed to fetch images:', imagesResponse.status, imagesResponse.statusText);
        toast.error('Failed to fetch images.');
      }

      // Assign signed_image_url to each character
      const charactersWithSignedUrls = await Promise.all(
        charactersData.map(async (character) => {
          if (character.image_id) {
            try {
              const signedUrlResponse = await fetch('/api/images/refresh-signed-url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageId: character.image_id }),
              });

              if (signedUrlResponse.ok) {
                const { newSignedUrl } = await signedUrlResponse.json();
                return { ...character, signed_image_url: newSignedUrl };
              } else {
                console.error(`Failed to fetch signed URL for character ID ${character.id}:`, signedUrlResponse.statusText);
                return { ...character, signed_image_url: '/default-avatar.png' }; // Fallback image
              }
            } catch (error) {
              console.error(`Error fetching signed URL for character ID ${character.id}:`, error);
              return { ...character, signed_image_url: '/default-avatar.png' }; // Fallback image
            }
          } else {
            console.warn(`Character ID ${character.id} does not have an associated image_id.`);
            return { ...character, signed_image_url: '/default-avatar.png' }; // Fallback image
          }
        })
      );

      setCharacters(charactersWithSignedUrls);
      setImages(imagesData);
    } catch (error) {
      console.error('Error fetching data:', error);
      toast.error('An error occurred while fetching data.');
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Handle creating a new character
  const handleCreateCharacter = async () => {
    if (!newCharacter.name || !newCharacter.image_id) {
      toast.error('Name and image are required.');
      return;
    }
  
    try {
      const response = await fetch('/api/chat/characters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newCharacter),
      });
  
      if (response.ok) {
        const data = await response.json();
        const createdCharacter: Character = data.character;
  
        // Check if character and image_id are present
        if (!createdCharacter || !createdCharacter.image_id) {
          console.error('Character data is missing after creation:', data);
          toast.error('Failed to retrieve character data.');
          return;
        }
  
        // Fetch signed URL for the newly created character
        try {
          const signedUrlResponse = await fetch('/api/images/refresh-signed-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imageId: createdCharacter.image_id }),
          });
  
          if (signedUrlResponse.ok) {
            const { newSignedUrl } = await signedUrlResponse.json();
            createdCharacter.signed_image_url = newSignedUrl;
          } else {
            console.error(`Failed to fetch signed URL for character ID ${createdCharacter.id}:`, signedUrlResponse.statusText);
            createdCharacter.signed_image_url = '/default-avatar.png'; // Fallback image
          }
        } catch (error) {
          console.error(`Error fetching signed URL for character ID ${createdCharacter.id}:`, error);
          createdCharacter.signed_image_url = '/default-avatar.png'; // Fallback image
        }
  
        // Update the state to include the new character
        setCharacters((prevCharacters) => [...prevCharacters, createdCharacter]);
        setIsDialogOpen(false);
        setNewCharacter({
          name: '',
          image_id: '',
          personality_traits: '',
          other_info: '',
        });
        toast.success('Character created successfully!');
      } else {
        const errorData = await response.json();
        console.error('Failed to create character:', errorData);
        toast.error(errorData.error || 'Failed to create character.');
      }
    } catch (error) {
      console.error('Error creating character:', error);
      toast.error('An error occurred while creating the character.');
    }
  };

  // Handle image click to enlarge
  const handleImageClick = (character: Character) => {
    setSelectedCharacter(character);
    setIsImageEnlarged(true);
  };

  return (
    <div className="w-full">
      {/* Header and New Character Button */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-white">Characters</h2>
        <Button
          text="New Character"
          className="bg-green-500 text-white px-4 py-2 rounded-lg"
          onClick={() => setIsDialogOpen(true)}
        />
      </div>

      {/* Character Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {characters.map((character) => (
          <div
            key={character.id}
            className="bg-gray-700 rounded-lg p-4 flex flex-col items-center cursor-pointer hover:bg-gray-600 transition duration-200"
            onClick={() => onSelectCharacter(character)}
          >
            <img
              src={character.signed_image_url || '/default-avatar.png'} // Fallback to default image
              alt={character.name}
              className="w-24 h-24 rounded-full object-cover mb-2"
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering the parent onClick
                handleImageClick(character);
              }}
            />
            <h3 className="text-xl text-white">{character.name}</h3>
          </div>
        ))}
      </div>

      {/* Create Character Dialog */}
      <Dialog isOpen={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <h2 className="text-2xl mb-4">Create New Character</h2>
        <div className="flex flex-col space-y-4">
          <input
            type="text"
            className="px-4 py-2 border rounded-lg"
            placeholder="Character Name"
            value={newCharacter.name}
            onChange={(e) => setNewCharacter({ ...newCharacter, name: e.target.value })}
            required
          />

          {/* Image Selection Dropdown */}
          <select
            className="px-4 py-2 border rounded-lg"
            value={newCharacter.image_id}
            onChange={(e) => setNewCharacter({ ...newCharacter, image_id: e.target.value })}
            required
          >
            <option value="">Select an Image</option>
            {images.map((image) => (
              <option key={image.id} value={image.id}>
                {image.filename}
              </option>
            ))}
          </select>

          <textarea
            className="px-4 py-2 border rounded-lg"
            placeholder="Personality Traits"
            value={newCharacter.personality_traits}
            onChange={(e) => setNewCharacter({ ...newCharacter, personality_traits: e.target.value })}
          />
          <textarea
            className="px-4 py-2 border rounded-lg"
            placeholder="Other Information"
            value={newCharacter.other_info}
            onChange={(e) => setNewCharacter({ ...newCharacter, other_info: e.target.value })}
          />
          <div className="flex justify-end space-x-4">
            <Button
              text="Cancel"
              className="bg-gray-300 text-black px-4 py-2 rounded-lg"
              onClick={() => setIsDialogOpen(false)}
            />
            <Button
              text="Create"
              className="bg-blue-500 text-white px-4 py-2 rounded-lg"
              onClick={handleCreateCharacter}
            />
          </div>
        </div>
      </Dialog>

      {/* Enlarge Character Image Modal */}
      {selectedCharacter && (
        <EnlargeCharacterImage
          character={selectedCharacter}
          isEnlarged={isImageEnlarged}
          onClose={() => setIsImageEnlarged(false)}
        />
      )}
    </div>
  );
};

export default CharacterList;
