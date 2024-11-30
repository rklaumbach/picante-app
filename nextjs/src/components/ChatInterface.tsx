// src/components/ChatInterface.tsx

'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Message } from '@/types/types';
import Button from './Button';
import { supabaseFrontendClient } from '@/lib/supabaseFrontendClient'; // Correct import
import { toast } from 'react-toastify';
import { RealtimePostgresInsertPayload } from '@supabase/supabase-js';

interface ChatInterfaceProps {
  chatId: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ chatId }) => {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Function to scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

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

  // Fetch messages
  const fetchMessages = async (chat_id: string) => {
    console.log(`Fetching messages for chatId: ${chat_id}`);
    try {
      const response = await fetch(`/api/chat/chats/messages?chat_id=${encodeURIComponent(chat_id)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Fetched messages:', data.messages);
        setMessages(data.messages);
        scrollToBottom();
      } else {
        console.error('Failed to fetch messages:', response.status, response.statusText);
        toast.error('Failed to fetch messages.');
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
      toast.error('An error occurred while fetching messages.');
    }
  };

  useEffect(() => {
    if (session) {
      fetchMessages(chatId);
    } else {
      console.log('Session not available. Skipping fetchMessages.');
    }

    const channelName = `chat-${chatId}`;
    console.log(`Subscribing to Supabase channel: ${channelName}`);

    // Subscribe to real-time updates using Supabase Realtime
    const subscription = supabaseFrontendClient
      .channel(channelName)
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'messages', filter: `chat_id=eq.${chatId}` },
        (payload: RealtimePostgresInsertPayload<Message>) => {
          console.log('New message received via Supabase:', payload.new);
          setMessages((prev) => [...prev, payload.new]);
          scrollToBottom();
        }
      )
      .subscribe();

    // Cleanup subscription on unmount
    return () => {
      console.log(`Unsubscribing from Supabase channel: ${channelName}`);
      supabaseFrontendClient.removeChannel(subscription);
    };
  }, [chatId, session]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    setIsSending(true);
    console.log('Sending message:', input.trim());

    try {
      const response = await fetch(`/api/chat/chats/messages?chat_id=${encodeURIComponent(chatId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: 'user', content: input.trim() }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Message sent successfully:', data);
        // The assistant's response will be automatically added via real-time subscription
        setInput('');
      } else {
        const errorData = await response.json();
        console.error('Failed to send message:', errorData);
        toast.error(errorData.error || 'Failed to send message.');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('An error occurred while sending the message.');
    } finally {
      setIsSending(false);
      scrollToBottom();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Debugging: Log messages state
  useEffect(() => {
    console.log('Current messages:', messages);
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`rounded-lg px-4 py-2 max-w-xs ${
                msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-800'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Box */}
      <div className="flex space-x-2 p-4 bg-gray-800">
        <textarea
          className="flex-1 px-4 py-2 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-black resize-none"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          rows={2}
        />
        <Button
          text={isSending ? 'Sending...' : 'Send'}
          className={`bg-blue-600 text-white px-4 py-2 rounded-lg ${
            isSending ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700'
          }`}
          onClick={handleSendMessage}
          disabled={isSending || !input.trim()}
        />
      </div>
    </div>
  );
};

export default ChatInterface;
