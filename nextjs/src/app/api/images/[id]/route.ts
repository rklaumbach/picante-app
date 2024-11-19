// src/app/api/images/[id]/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { createClient } from '@supabase/supabase-js';

export async function DELETE(req: NextRequest, { params }: { params: { id: string } }) {
  const imageId = params.id;

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

    // Delete the image from Supabase storage
    const { error: deleteError } = await supabase
      .storage
      .from('user_images')
      .remove([imageData.image_path]);

    if (deleteError) {
      console.error('Error deleting image from storage:', deleteError);
      return NextResponse.json({ error: 'Failed to delete image from storage.' }, { status: 500 });
    }

    // Delete the image record from the database
    const { error: dbDeleteError } = await supabase
      .from('images')
      .delete()
      .eq('id', imageId);

    if (dbDeleteError) {
      console.error('Error deleting image from database:', dbDeleteError);
      return NextResponse.json({ error: 'Failed to delete image record.' }, { status: 500 });
    }

    return NextResponse.json({ message: 'Image deleted successfully.' }, { status: 200 });
  } catch (error) {
    console.error('Error in DELETE /api/images/[id]:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
