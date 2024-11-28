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
    const { data, error } = await supabaseAdmin
      .from('chats')
      .select(`
        *,
        characters(id, name, signed_image_url)
      `)
      .eq('user_id', userId)
      .order('updated_at', { ascending: false });

    if (error) {
      console.error('Error fetching chats:', error);
      return NextResponse.json({ error: 'Failed to fetch chats.' }, { status: 500 });
    }

    return NextResponse.json({ chats: data }, { status: 200 });
  } catch (error) {
    console.error('Error in GET /api/chat/chats:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
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
  