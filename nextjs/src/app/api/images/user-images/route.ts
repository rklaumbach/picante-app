// src/app/api/images/user-images/route.ts

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

    // Fetch images associated with the user
    const { data, error } = await supabase
      .from('images')
      .select('id, filename')
      .eq('user_id', userId)
      .order('created_at', { ascending: true });

    if (error) {
      console.error('Error fetching images:', error);
      return NextResponse.json({ error: 'Failed to fetch images.' }, { status: 500 });
    }

    return NextResponse.json({ images: data }, { status: 200 });
  } catch (error) {
    console.error('Error in GET /api/images/user-images:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
