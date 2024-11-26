// src/app/api/images/refresh-signed-url/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { createClient } from '@supabase/supabase-js';

export async function POST(req: NextRequest) {
  try {
    const { imageId } = await req.json();

    if (!imageId) {
      return NextResponse.json({ error: 'Image ID is required.' }, { status: 400 });
    }

    // Authenticate the user
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Initialize Supabase client with Service Role Key
    const supabase = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Fetch the image to verify ownership
    const { data: imageData, error: fetchError } = await supabase
      .from('images')
      .select('*')
      .eq('id', imageId)
      .single();

    if (fetchError || !imageData) {
      console.error('Error fetching image:', fetchError);
      return NextResponse.json({ error: 'Image not found.' }, { status: 404 });
    }

    if (imageData.user_id !== userId) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    // Generate a new signed URL
    const { data: signedData, error: signedError } = await supabase
      .storage
      .from('user-images')
      .createSignedUrl(imageData.image_path, 60 * 60); // URL valid for 1 hour

    if (signedError || !signedData) {
      console.error('Error generating signed URL:', signedError);
      return NextResponse.json({ error: 'Failed to generate signed URL.' }, { status: 500 });
    }

    return NextResponse.json({ newSignedUrl: signedData.signedUrl }, { status: 200 });
  } catch (error) {
    console.error('Error in /api/images/refresh-signed-url:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
