// src/app/api/chat/chats/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';
import { getToken } from 'next-auth/jwt';

export async function GET(req: NextRequest) {
  try {
    // Authenticate the user
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token) {
      console.error('Authentication failed: No token found.');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Fetch chats along with associated characters
    const { data: chats, error } = await supabaseAdmin
      .from('chats')
      .select(`
        *,
        characters:characters(id, user_id, name, image_path, personality_traits, other_info)
      `)
      .eq('user_id', userId)
      .order('updated_at', { ascending: false });

    if (error) {
      console.error('Supabase fetch error:', error);
      return NextResponse.json(
        { error: 'Failed to fetch chats.', details: error.message },
        { status: 500 }
      );
    }

    // Generate signed_image_url for each character
    const chatsWithCharacterURLs = await Promise.all(
      chats.map(async (chat) => {
        if (chat.characters && chat.characters.image_path) {
          const { data: signedData, error: urlError } = await supabaseAdmin.storage
            .from('characters') // Ensure this matches your storage bucket name
            .createSignedUrl(chat.characters.image_path, 60*60); // URL valid for 60 seconds

          if (urlError) {
            console.error(`Error generating signed URL for character ${chat.characters.id}:`, urlError);
            chat.characters.signed_image_url = null; // Or set a default image URL
          } else {
            chat.characters.signed_image_url = signedData?.signedUrl;
          }
        } else {
          chat.characters = {
            ...chat.characters,
            signed_image_url: null, // Or set a default image URL
          };
        }

        return chat;
      })
    );

    return NextResponse.json({ chats: chatsWithCharacterURLs }, { status: 200 });
  } catch (error: any) {
    console.error('Unexpected error in GET /api/chat/chats:', error);
    return NextResponse.json(
      { error: 'Internal Server Error', details: error.message },
      { status: 500 }
    );
  }
}

// src/app/api/chat/chats/route.ts

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
  
      // Insert the new chat into the database
      const { data, error } = await supabaseAdmin
        .from('chats')
        .insert([
          {
            user_id: userId,
            title,
            scenario,
            character_id,
          },
        ])
        .single();
  
      if (error) {
        console.error('Error creating chat:', error);
        return NextResponse.json({ error: 'Failed to create chat.' }, { status: 500 });
      }
  
      return NextResponse.json({ chat: data }, { status: 201 });
    } catch (error) {
      console.error('Error in POST /api/chat/chats:', error);
      return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
  }
  