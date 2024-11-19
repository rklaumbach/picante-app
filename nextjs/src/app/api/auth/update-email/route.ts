// src/app/api/auth/update-email/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';

interface UpdateEmailRequestBody {
  email: string;
}

export async function PUT(req: NextRequest) {
  try {
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { email } = (await req.json()) as UpdateEmailRequestBody;

    if (!email) {
      return NextResponse.json({ error: 'Email is required.' }, { status: 400 });
    }

    const userId = token.sub as string;

    // Update user's email in Supabase Auth
    const { data, error } = await supabaseAdmin.auth.admin.updateUserById(userId, {
      email,
      email_confirm: false, // Require email confirmation after update
    });

    if (error || !data) {
      console.error('Error updating email:', error);
      return NextResponse.json({ error: 'Failed to update email.' }, { status: 500 });
    }

    // Optionally, you can send a confirmation email here if not handled by Supabase

    return NextResponse.json({ message: 'Email updated successfully. Please confirm your new email.' }, { status: 200 });
  } catch (error) {
    console.error('Error in /api/user/update-email:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
