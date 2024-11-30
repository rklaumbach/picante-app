// src/app/chat/[chat_id]/page.tsx

'use client';

import React, { useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Header from '../../../components/Header';
import ChatInterface from '../../../components/ChatInterface';
import BottomNav from '../../../components/BottomNav';
import { useSession } from 'next-auth/react';
import { toast } from 'react-toastify';

const ChatPage: React.FC = () => {
  const router = useRouter();
  const params = useParams();
  const chatIdParam = params.chat_id;

  // Ensure chatId is a string
  const chatId: string | undefined = Array.isArray(chatIdParam) ? chatIdParam[0] : chatIdParam;

  const { data: session, status } = useSession();

  // Debugging: Log session status and data
  useEffect(() => {
    console.log('Session status:', status);
    console.log('Session data:', session);
    if (status === 'loading') {
      console.log('Session is loading...');
      return; // Do nothing while loading
    }
    if (!session) {
      console.log('No session detected. Redirecting to /chat');
      toast.error('You must be logged in to access this chat.');
      router.push('/chat'); // Redirect to main chat page
    }
  }, [session, status, router]);

  // Redirect if no chatId is present
  useEffect(() => {
    if (!chatId) {
      console.log('No chat_id found in route. Redirecting to /chat');
      router.push('/chat');
    }
  }, [chatId, router]);

  if (!chatId) {
    // Optionally, you can render a loading state or null
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
