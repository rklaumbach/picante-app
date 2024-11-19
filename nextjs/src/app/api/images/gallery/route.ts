// src/app/api/images/gallery/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

interface Image {
  id: string;
  image_url: string;
  filename: string;
  body_prompt: string;
  face_prompt: string;
  resolution: string;
  created_at: string;
}

export async function GET(req: NextRequest) {
  try {
    // Authenticate the user
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Fetch images for the user
    const { data, error } = await supabase
      .from('images')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching images from Supabase:', error);
      return NextResponse.json(
        { error: 'Failed to fetch images.' },
        { status: 500 }
      );
    }

    // Map data to Image interface
    const images: Image[] = data.map((img) => ({
      id: img.id,
      image_url: img.image_url,
      filename: img.filename,
      body_prompt: img.body_prompt,
      face_prompt: img.face_prompt,
      resolution: img.resolution,
      created_at: img.created_at,
    }));

    return NextResponse.json({ images }, { status: 200 });
  } catch (error) {
    console.error('Error in /api/images/gallery:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
