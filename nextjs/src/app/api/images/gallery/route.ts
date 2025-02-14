// src/app/api/images/gallery/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { createClient } from '@supabase/supabase-js';


export async function GET(req: NextRequest) {
  try {
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

    // Generate signed URLs for each image
    const signedImages = await Promise.all(
      data.map(async (img: any) => {
        const { data: signedData, error: signedError } = await supabase
          .storage
          .from('user-images')
          .createSignedUrl(img.image_path, 60 * 60); // URL valid for 1 hour

        if (signedError) {
          console.error(`Error generating signed URL for ${img.image_path}:`, signedError);
          return null;
        }

        return {
          id: img.id,
          image_url: signedData?.signedUrl || '',
          filename: img.filename,
          body_prompt: img.body_prompt,
          face_prompt: img.face_prompt,
          width: img.width,
          height: img.height,
          created_at: img.created_at,
        };
      })
    );

    // Filter out any null entries due to errors
    const validImages = signedImages.filter((img) => img !== null);

    // Create the response with Cache-Control headers
    const response = NextResponse.json(
      { images: validImages },
      { status: 200 }
    );

    // Set Cache-Control headers
    response.headers.set('Cache-Control', 'public, max-age=3600, stale-while-revalidate=300');
    // Ensure caching varies per user to prevent serving URLs across users
    response.headers.set('Vary', 'Cookie');

    return response;
  } catch (error) {
    console.error('Error in /api/images/gallery:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
