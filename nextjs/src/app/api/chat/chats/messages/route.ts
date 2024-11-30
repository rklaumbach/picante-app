// src/app/api/chat/chats/messages/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '../../../../../lib/supabaseAdminClient';
import { getToken } from 'next-auth/jwt';

interface ChatJob {
  prompt: string;
  history: { role: string; content: string }[];
  max_length: number;
  temperature: number;
  personality_traits: string;
  other_info: string;
  context: string;
}

interface ChatResponse {
  response: string;
  history: { role: string; content: string }[];
}

// src/app/api/chat/chats/messages/route.ts

export async function POST(req: NextRequest) {
  // Extract 'chat_id' from query parameters
  const { searchParams } = req.nextUrl;
  const chat_id = searchParams.get('chat_id');

  if (!chat_id) {
    return NextResponse.json({ error: 'chat_id is required as a query parameter.' }, { status: 400 });
  }

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
      .select('id, character_id, context') // Include 'context'
      .eq('id', chat_id)
      .eq('user_id', userId)
      .single();

    if (chatError || !chatData) {
      console.error('Error fetching chat:', chatError);
      return NextResponse.json({ error: 'Chat not found.' }, { status: 404 });
    }

    // Fetch character details for personality_traits and other_info
    const { data: characterData, error: characterError } = await supabaseAdmin
      .from('characters')
      .select('personality_traits, other_info')
      .eq('id', chatData.character_id)
      .single();

    if (characterError || !characterData) {
      console.error('Error fetching character details:', characterError);
      return NextResponse.json({ error: 'Character details not found.' }, { status: 500 });
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
      prompt: content,
      history: history,
      max_length: 200, // Adjust as needed
      temperature: 0.7, // Adjust as needed
      personality_traits: characterData.personality_traits,
      other_info: characterData.other_info,
      context: chatData.context,
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
    console.error('Error in POST /api/chat/chats/messages:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// Helper function to communicate with Modal Chat Service
const sendToModalChat = async (job: ChatJob): Promise<ChatResponse> => {
  const modalChatUrl = process.env.MODAL_CHAT_API_URL as string;
  const chatEndpoint = `${modalChatUrl}/chat`; // Append /chat to the base URL

  const response = await fetch(chatEndpoint, {
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
