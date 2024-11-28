// src/app/chat/[chat_id]/page.tsx

'use client';

import React from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Header from '../../../components/Header';
import ChatInterface from '../../../components/ChatInterface';
import BottomNav from '../../../components/BottomNav';

const ChatPage: React.FC = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const chatId = searchParams.get('chat_id');

  if (!chatId) {
    // Redirect to chat list if no chat_id is present
    router.push('/chat');
    return null;
  }

  return (
    <>
      <main className="flex flex-col items-center px-4 pb-20 mx-auto w-full max-w-7xl min-h-screen pt-24">
        <div className="app-container flex flex-col items-center w-full">
          <Header title="Chat" />
          <div className="w-full mt-8 h-[70vh]">
            <ChatInterface chatId={chatId} />
          </div>
        </div>
      </main>
      <BottomNav />
    </>
  );
};

export default ChatPage;
