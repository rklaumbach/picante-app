// src/app/api/user/credits/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';


interface UserCreditsResponse {
  credits: number;
}


export async function GET(req: NextRequest) {
  try {
    // Retrieve the token using NextAuth.js
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Fetch user credits from Supabase
    const { data, error } = await supabaseAdmin
      .from('user_profiles')
      .select('credits')
      .eq('id', userId)
      .single();

    if (error || !data) {
      console.error('Supabase Error:', error); // Enhanced logging
      return NextResponse.json({ error: 'Failed to retrieve credits.' }, { status: 500 });
    }

    const credits = data.credits;

    return NextResponse.json({ credits }, { status: 200 });
  } catch (error) {
    console.error('Error in /api/user/credits:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
