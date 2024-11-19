// src/app/api/auth/reset-password/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';

interface ResetPasswordRequestBody {
  email: string;
}

export async function POST(req: NextRequest) {
  try {
    const { email } = (await req.json()) as ResetPasswordRequestBody;

    if (!email) {
      return NextResponse.json({ error: 'Email is required.' }, { status: 400 });
    }

    // Trigger password reset email
    const { data, error } = await supabaseAdmin.auth.resetPasswordForEmail(email, {
      redirectTo: `${process.env.SITE_URL}/update-password`, // Redirect URL after resetting password
    });

    if (error) {
      console.error('Error sending password reset email:', error);
      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    return NextResponse.json({ message: 'Password reset email sent.' }, { status: 200 });
  } catch (error) {
    console.error('Error in /api/auth/reset-password:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
