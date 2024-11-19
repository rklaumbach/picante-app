// src/app/api/user/profile/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';

interface UserProfile {
  id: string;
  email: string;
  credits: number;
  subscriptionStatus: string; // Changed from subscription_status
  // Include other fields if necessary
}

const JWT_SECRET = process.env.NEXTAUTH_SECRET!;

export async function GET(req: NextRequest) {
  try {
    const token = await getToken({ req, secret: JWT_SECRET });

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const userId = token.sub as string;

    // Use Admin API to get user details
    const { data: userData, error: userError } = await supabaseAdmin.auth.admin.getUserById(userId);

    console.log(userData);

    if (userError || !userData) {
      console.error('Error fetching user from auth API:', userError);
      return NextResponse.json({ error: 'Failed to retrieve user email.' }, { status: 500 });
    }

    // Fetch profile data from user_profiles
    const { data: profileData, error: profileError } = await supabaseAdmin
      .from('user_profiles')
      .select('*')
      .eq('id', userId)
      .single();

    if (profileError || !profileData) {
      console.error('Error fetching user profile:', profileError);
      return NextResponse.json({ error: 'Failed to retrieve user profile.' }, { status: 500 });
    }

    const userProfile: UserProfile = {
      id: profileData.id,
      email: userData.user.email || '', // Ensure email is always a string
      credits: profileData.credits,
      subscriptionStatus: profileData.subscription_status, // Align with client-side interface
      // Map other fields as necessary
    };

    return NextResponse.json({ user: userProfile }, { status: 200 });
  } catch (error) {
    console.error('Error in /api/user/profile GET:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// Similarly, update the PUT method if necessary
