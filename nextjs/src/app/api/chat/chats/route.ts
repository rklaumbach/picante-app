// src/app/api/chat/chats/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';
import { getToken } from 'next-auth/jwt';

export async function GET(req: NextRequest) {
  try {
    // Authenticate the user
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Fetch chats associated with the user
    const { data: chats, error: chatsError } = await supabaseAdmin
      .from('chats')
      .select(`
        id,
        title,
        scenario,
        updated_at,
        character:characters (
          id,
          name,
          image_id
        )
      `)
      .eq('user_id', userId) // Filter chats by user
      .order('updated_at', { ascending: false });

    if (chatsError) {
      console.error('Error fetching chats:', chatsError);
      return NextResponse.json({ error: 'Failed to fetch chats.' }, { status: 500 });
    }

    return NextResponse.json({ chats }, { status: 200 });
  } catch (error) {
    console.error('Error in GET /api/chat/chats:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    // Authenticate the user
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Parse the request body
    const { title, scenario, character_id } = await req.json();

    if (!title || !scenario || !character_id) {
      return NextResponse.json({ error: 'Title, scenario, and character_id are required.' }, { status: 400 });
    }

    // Verify that the character belongs to the user
    const { data: characterData, error: characterError } = await supabaseAdmin
      .from('characters')
      .select('id, user_id, image_id')
      .eq('id', character_id)
      .single();

    if (characterError || !characterData) {
      console.error('Error fetching character:', characterError);
      return NextResponse.json({ error: 'Character not found.' }, { status: 404 });
    }

    if (characterData.user_id !== userId) {
      return NextResponse.json({ error: 'Forbidden: Character does not belong to the user.' }, { status: 403 });
    }

    // Insert the new chat into the database
    const { data: chatData, error: chatError } = await supabaseAdmin
      .from('chats')
      .insert([
        {
          title,
          scenario,
          character_id,
          user_id: userId, // Assuming 'user_id' exists in 'chats' table
        },
      ])
      .select()
      .single();

    if (chatError) {
      console.error('Error creating chat:', chatError);
      return NextResponse.json({ error: 'Failed to create chat.' }, { status: 500 });
    }

    return NextResponse.json({ chat: chatData }, { status: 201 });
  } catch (error) {
    console.error('Error in POST /api/chat/chats:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
