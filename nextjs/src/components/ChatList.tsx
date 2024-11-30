// src/components/ChatList.tsx

'use client';

import React, { useEffect, useState } from 'react';
import { Chat, Character } from '@/types/types';
import Button from './Button';
import Dialog from './Dialog';
import { useRouter } from 'next/navigation';
import { toast } from 'react-toastify';

interface ChatListProps {
  selectedCharacter: Character;
}

const ChatList: React.FC<ChatListProps> = ({ selectedCharacter }) => {
  const [chats, setChats] = useState<Chat[]>([]);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newChat, setNewChat] = useState({
    title: '',
    scenario: '',
  });
  const router = useRouter();

  const fetchChats = async () => {
    try {
      const response = await fetch('/api/chat/chats', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        const data = await response.json();
        const fetchedChats: Chat[] = data.chats;

        if (!fetchedChats) {
          toast.error('No chats data received.');
          return;
        }

        // Extract unique image_ids from chats
        const uniqueImageIds = Array.from(
          new Set(fetchedChats.map((chat) => chat.character.image_id))
        );

        // Fetch signed URLs for unique image_ids
        const signedUrlPromises = uniqueImageIds.map(async (imageId) => {
          try {
            const res = await fetch('/api/images/refresh-signed-url', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ imageId }),
            });

            if (res.ok) {
              const { newSignedUrl } = await res.json();
              return { imageId, signedUrl: newSignedUrl };
            } else {
              console.error(`Failed to fetch signed URL for image ID ${imageId}:`, res.statusText);
              return { imageId, signedUrl: '/default-avatar.png' };
            }
          } catch (error) {
            console.error(`Error fetching signed URL for image ID ${imageId}:`, error);
            return { imageId, signedUrl: '/default-avatar.png' };
          }
        });

        const signedUrls = await Promise.all(signedUrlPromises);
        const imageIdToUrlMap: { [key: string]: string } = {};
        signedUrls.forEach(({ imageId, signedUrl }) => {
          imageIdToUrlMap[imageId] = signedUrl;
        });

        // Assign signed_image_url to each chat's character
        const chatsWithImages = fetchedChats.map((chat) => ({
          ...chat,
          character: {
            ...chat.character,
            signed_image_url: imageIdToUrlMap[chat.character.image_id] || '/default-avatar.png',
          },
        }));

        setChats(chatsWithImages);
      } else {
        const errorData = await response.json();
        console.error('Failed to fetch chats:', errorData);
        toast.error(errorData.error || 'Failed to fetch chats.');
      }
    } catch (error) {
      console.error('Error fetching chats:', error);
      toast.error('An error occurred while fetching chats.');
    }
  };

  useEffect(() => {
    fetchChats();
  }, []);

  const handleCreateChat = async () => {
    if (!newChat.title || !newChat.scenario) {
      toast.error('Title and scenario are required.');
      return;
    }

    try {
      const response = await fetch('/api/chat/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: newChat.title,
          scenario: newChat.scenario,
          character_id: selectedCharacter.id,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const createdChat: Chat = data.chat;

        // Defensive check to ensure 'character' exists
        if (!createdChat || !createdChat.character) {
          console.error('Created chat does not have a character:', data);
          toast.error('Failed to retrieve chat details.');
          return;
        }

        // Fetch signed URL for the created chat's character
        try {
          const res = await fetch('/api/images/refresh-signed-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imageId: createdChat.character.image_id }),
          });

          let signedUrl = '/default-avatar.png';
          if (res.ok) {
            const { newSignedUrl } = await res.json();
            signedUrl = newSignedUrl;
          } else {
            console.error(`Failed to fetch signed URL for image ID ${createdChat.character.image_id}:`, res.statusText);
          }

          const chatWithImage = {
            ...createdChat,
            character: {
              ...createdChat.character,
              signed_image_url: signedUrl,
            },
          };

          setChats([chatWithImage, ...chats]);
          setIsDialogOpen(false);
          setNewChat({ title: '', scenario: '' });
          toast.success('Chat created successfully!');
        } catch (error) {
          console.error('Error fetching signed URL for created chat:', error);
          // Even if signed URL fetch fails, add the chat with fallback image
          const chatWithFallbackImage = {
            ...createdChat,
            character: {
              ...createdChat.character,
              signed_image_url: '/default-avatar.png',
            },
          };
          setChats([chatWithFallbackImage, ...chats]);
          setIsDialogOpen(false);
          setNewChat({ title: '', scenario: '' });
          toast.success('Chat created successfully, but failed to load image.');
        }
      } else {
        const errorData = await response.json();
        console.error('Failed to create chat:', errorData);
        toast.error(errorData.error || 'Failed to create chat.');
      }
    } catch (error) {
      console.error('Error creating chat:', error);
      toast.error('An error occurred while creating the chat.');
    }
  };

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-white">Chats</h2>
        <Button
          text="New Chat"
          className="bg-green-500 text-white px-4 py-2 rounded-lg"
          onClick={() => setIsDialogOpen(true)}
        />
      </div>
      <div className="space-y-4">
        {chats.length === 0 ? (
          <p className="text-white">No chats available. Start a new conversation!</p>
        ) : (
          chats.map((chat) => (
            <div
              key={chat.id}
              className="bg-gray-700 rounded-lg p-4 flex items-center justify-between cursor-pointer hover:bg-gray-600 transition duration-200"
              onClick={() => router.push(`/chat/${chat.id}`)}
            >
              <div className="flex items-center space-x-4">
                {/* Ensure chat.character exists before accessing properties */}
                {chat.character && (
                  <img
                    src={chat.character.signed_image_url || '/default-avatar.png'}
                    alt={chat.character.name}
                    className="w-12 h-12 rounded-full object-cover"
                  />
                )}
                <div>
                  <h3 className="text-xl text-white">{chat.title}</h3>
                  <p className="text-gray-300">{chat.scenario}</p>
                </div>
              </div>
              <p className="text-gray-400 text-sm">{new Date(chat.updated_at).toLocaleString()}</p>
            </div>
          ))
        )}
      </div>

      {/* Create Chat Dialog */}
      <Dialog isOpen={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <h2 className="text-2xl mb-4">Create New Chat</h2>
        <div className="flex flex-col space-y-4">
          <input
            type="text"
            className="px-4 py-2 border rounded-lg"
            placeholder="Chat Title"
            value={newChat.title}
            onChange={(e) => setNewChat({ ...newChat, title: e.target.value })}
            required
          />
          <textarea
            className="px-4 py-2 border rounded-lg"
            placeholder="Chat Scenario"
            value={newChat.scenario}
            onChange={(e) => setNewChat({ ...newChat, scenario: e.target.value })}
            required
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
              onClick={handleCreateChat}
            />
          </div>
        </div>
      </Dialog>
    </div>
  );
};

export default ChatList;
