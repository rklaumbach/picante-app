// src/app/chat/page.tsx

'use client';

import React, { useState } from 'react';
import Header from '../../components/Header';
import ChatList from '../../components/ChatList';
import CharacterList from '../../components/CharacterList';
import BottomNav from '../../components/BottomNav';
import { Character } from '@/types/types';
import { useRouter } from 'next/navigation';
import Button from '../../components/Button';

const ChatPage: React.FC = () => {
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const router = useRouter();

  const handleSelectCharacter = (character: Character) => {
    setSelectedCharacter(character);
    // Optionally, redirect to create a new chat with the selected character
  };

  return (
    <>
      <main className="flex flex-col items-center px-4 pb-20 mx-auto w-full max-w-7xl min-h-screen pt-24">
        <div className="app-container flex flex-col items-center w-full">
          <Header title="Chats" />

          {/* Character Selection */}
          <CharacterList onSelectCharacter={handleSelectCharacter} />

          {/* Chat List */}
          {selectedCharacter && (
            <div className="w-full mt-8">
              <ChatList selectedCharacter={selectedCharacter} />
            </div>
          )}
        </div>
      </main>
      <BottomNav />
    </>
  );
};

export default ChatPage;
