// src/app/api/chat/characters/route.ts

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

    // Fetch characters associated with the user
    const { data, error } = await supabaseAdmin
      .from('characters')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: true });

    if (error) {
      console.error('Error fetching characters:', error);
      return NextResponse.json({ error: 'Failed to fetch characters.' }, { status: 500 });
    }

    return NextResponse.json({ characters: data }, { status: 200 });
  } catch (error) {
    console.error('Error in GET /api/chat/characters:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// src/app/api/chat/characters/route.ts

export async function POST(req: NextRequest) {
    try {
      // Authenticate the user
      const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
      if (!token) {
        return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
      }
  
      const userId = token.sub as string;
  
      // Parse the request body
      const { name, image_id, personality_traits, other_info } = await req.json();
  
      if (!name || !image_id) {
        return NextResponse.json({ error: 'Name and image path are required.' }, { status: 400 });
      }
  
      // Insert the new character into the database
      const { data, error } = await supabaseAdmin
        .from('characters')
        .insert([
          {
            user_id: userId,
            name,
            image_id,
            personality_traits,
            other_info,
          },
        ])
        .single();
  
      if (error) {
        console.error('Error creating character:', error);
        return NextResponse.json({ error: 'Failed to create character.' }, { status: 500 });
      }
  
      return NextResponse.json({ character: data }, { status: 201 });
    } catch (error) {
      console.error('Error in POST /api/chat/characters:', error);
      return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
  }
  