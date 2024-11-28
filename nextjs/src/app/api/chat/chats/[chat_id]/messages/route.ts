// src/app/api/chat/chats/[chat_id]/messages/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '../../../../../../lib/supabaseAdminClient';
import { getToken } from 'next-auth/jwt';

interface ChatJob {
  prompt: string;
  history: { role: string; content: string }[];
  max_length?: number;
  temperature?: number;
}

interface ChatResponse {
  response: string;
  history: { role: string; content: string }[];
}

const sendToModalChat = async (job: ChatJob): Promise<ChatResponse> => {
  const response = await fetch(process.env.MODAL_CHAT_API_URL as string, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      // Add any necessary authentication headers here
    },
    body: JSON.stringify(job),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Failed to communicate with Modal Chat Service.');
  }

  const data: ChatResponse = await response.json();
  return data;
};


export async function GET(
  req: NextRequest,
  context: { params: { chat_id: string }; searchParams: URLSearchParams }
) {
  const { chat_id } = context.params;

  try {
    // Authenticate the user
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Verify that the chat belongs to the user
    const { data: chatData, error: chatError } = await supabaseAdmin
      .from('chats')
      .select('id')
      .eq('id', chat_id)
      .eq('user_id', userId)
      .single();

    if (chatError || !chatData) {
      console.error('Error fetching chat:', chatError);
      return NextResponse.json({ error: 'Chat not found.' }, { status: 404 });
    }

    // Fetch messages for the chat
    const { data, error } = await supabaseAdmin
      .from('messages')
      .select('*')
      .eq('chat_id', chat_id)
      .order('timestamp', { ascending: true });

    if (error) {
      console.error('Error fetching messages:', error);
      return NextResponse.json({ error: 'Failed to fetch messages.' }, { status: 500 });
    }

    return NextResponse.json({ messages: data }, { status: 200 });
  } catch (error) {
    console.error('Error in GET /api/chat/chats/[chat_id]/messages:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// src/app/api/chat/chats/[chat_id]/messages/route.ts

export async function POST(
  req: NextRequest,
  context: { params: { chat_id: string }; searchParams: URLSearchParams }
) {
  const { chat_id } = context.params;

  try {
    // Authenticate the user
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Verify that the chat belongs to the user
    const { data: chatData, error: chatError } = await supabaseAdmin
      .from('chats')
      .select('id, character_id')
      .eq('id', chat_id)
      .eq('user_id', userId)
      .single();

    if (chatError || !chatData) {
      console.error('Error fetching chat:', chatError);
      return NextResponse.json({ error: 'Chat not found.' }, { status: 404 });
    }

    // Parse the request body
    const { role, content } = await req.json();

    if (!role || !content) {
      return NextResponse.json({ error: 'Role and content are required.' }, { status: 400 });
    }

    if (role !== 'user') {
      return NextResponse.json({ error: 'Invalid role. Only user messages are allowed here.' }, { status: 400 });
    }

    // Insert the user's message into the database
    const { data: userMessage, error: userInsertError } = await supabaseAdmin
      .from('messages')
      .insert([
        {
          chat_id,
          role,
          content,
        },
      ])
      .single();

    if (userInsertError) {
      console.error('Error sending user message:', userInsertError);
      return NextResponse.json({ error: 'Failed to send message.' }, { status: 500 });
    }

    // Fetch the updated message history
    const { data: messages, error: fetchMessagesError } = await supabaseAdmin
      .from('messages')
      .select('*')
      .eq('chat_id', chat_id)
      .order('timestamp', { ascending: true });

    if (fetchMessagesError) {
      console.error('Error fetching message history:', fetchMessagesError);
      return NextResponse.json({ error: 'Failed to retrieve message history.' }, { status: 500 });
    }

    // Prepare the prompt and history for Modal Chat
    const prompt = content;
    const history = messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));

    // Communicate with Modal Chat Service
    const chatResponse = await sendToModalChat({
      prompt,
      history,
      max_length: 200, // Adjust as needed
      temperature: 0.7, // Adjust as needed
    });

    // Insert the assistant's response into the database
    const { data: assistantMessage, error: assistantInsertError } = await supabaseAdmin
      .from('messages')
      .insert([
        {
          chat_id,
          role: 'assistant',
          content: chatResponse.response,
        },
      ])
      .single();

    if (assistantInsertError) {
      console.error('Error inserting assistant message:', assistantInsertError);
      return NextResponse.json({ error: 'Failed to process assistant response.' }, { status: 500 });
    }

    // Optionally, update the chat's updated_at timestamp
    const { error: updateChatError } = await supabaseAdmin
      .from('chats')
      .update({ updated_at: new Date().toISOString() })
      .eq('id', chat_id);

    if (updateChatError) {
      console.error('Error updating chat timestamp:', updateChatError);
      // Not critical, so we won't return an error to the client
    }

    // Return the assistant's response to the frontend
    return NextResponse.json({ assistant: assistantMessage }, { status: 200 });
  } catch (error) {
    console.error('Error in POST /api/chat/chats/[chat_id]/messages:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}