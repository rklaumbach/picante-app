// src/components/CharacterList.tsx

'use client';

import React, { useEffect, useState } from 'react';
import { Character } from '@/types/types';
import Button from './Button';
import Dialog from './Dialog';
import EnlargeCharacterImage from './EnlargeCharacterImage';
import { getSignedUrl } from '../lib/signedUrls';
import { toast } from 'react-toastify';

interface CharacterListProps {
  onSelectCharacter: (character: Character) => void;
}

const CharacterList: React.FC<CharacterListProps> = ({ onSelectCharacter }) => {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newCharacter, setNewCharacter] = useState({
    name: '',
    image_path: '',
    personality_traits: '',
    other_info: '',
  });
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const [isImageEnlarged, setIsImageEnlarged] = useState(false);

  const fetchCharacters = async () => {
    try {
      const response = await fetch('/api/chat/characters', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        const data = await response.json();
        setCharacters(data.characters);
      } else {
        console.error('Failed to fetch characters.');
        toast.error('Failed to fetch characters.');
      }
    } catch (error) {
      console.error('Error fetching characters:', error);
      toast.error('An error occurred while fetching characters.');
    }
  };

  useEffect(() => {
    fetchCharacters();
  }, []);

  const handleCreateCharacter = async () => {
    if (!newCharacter.name || !newCharacter.image_path) {
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
        setCharacters([...characters, data.character]);
        setIsDialogOpen(false);
        setNewCharacter({
          name: '',
          image_path: '',
          personality_traits: '',
          other_info: '',
        });
        toast.success('Character created successfully!');
      } else {
        const errorData = await response.json();
        toast.error(errorData.error || 'Failed to create character.');
      }
    } catch (error) {
      console.error('Error creating character:', error);
      toast.error('An error occurred while creating the character.');
    }
  };

  const handleImageClick = (character: Character) => {
    setSelectedCharacter(character);
    setIsImageEnlarged(true);
  };

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-white">Characters</h2>
        <Button
          text="New Character"
          className="bg-green-500 text-white px-4 py-2 rounded-lg"
          onClick={() => setIsDialogOpen(true)}
        />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {characters.map((character) => (
          <div
            key={character.id}
            className="bg-gray-700 rounded-lg p-4 flex flex-col items-center cursor-pointer"
            onClick={() => onSelectCharacter(character)}
          >
            <img
              src={character.signed_image_url}
              alt={character.name}
              className="w-24 h-24 rounded-full object-cover mb-2"
              onClick={(e) => {
                e.stopPropagation();
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
          <input
            type="text"
            className="px-4 py-2 border rounded-lg"
            placeholder="Image Path (storage/path.jpg)"
            value={newCharacter.image_path}
            onChange={(e) => setNewCharacter({ ...newCharacter, image_path: e.target.value })}
            required
          />
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
