// src/app/api/auth/register/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';
import { sgMail } from '../../../../lib/sendGrid'; // Adjust the import path as needed
import bcrypt from 'bcryptjs';

interface RegisterRequestBody {
  email: string;
  password: string;
}

export async function POST(req: NextRequest) {
  try {
    const { email, password } = (await req.json()) as RegisterRequestBody;

    // Validate input
    if (!email || !password) {
      return NextResponse.json({ error: 'Email and password are required.' }, { status: 400 });
    }

    // Check if user already exists in auth.users
    const { data: existingUser, error: fetchError } = await supabaseAdmin
      .from('auth.users')
      .select('id')
      .eq('email', email)
      .single();

    if (existingUser) {
      return NextResponse.json({ error: 'User already exists.' }, { status: 400 });
    }

    // Create user via Supabase Auth
    const { data: createData, error: createError } = await supabaseAdmin.auth.admin.createUser({
      email,
      password,
      email_confirm: false, // Automatically confirm the email
    });

    if (createError) {
      console.error('Error creating user:', createError);
      return NextResponse.json({ error: 'Failed to register user.' }, { status: 500 });
    }

    // **Important Correction: Extract userId correctly**
    // The response from createUser has the user object inside data.user
    const userId = createData?.id || createData?.user?.id;

    if (!userId) {
      console.error('Failed to retrieve user ID from createUser response:', createData);
      return NextResponse.json({ error: 'Failed to retrieve user ID.' }, { status: 500 });
    }

    // Log the userId for debugging
    console.log('Created userId:', userId);

    // Insert into user_profiles
    const { error: insertError } = await supabaseAdmin
      .from('user_profiles')
      .insert([
        {
          id: userId, // Ensure this matches the column name in user_profiles
          credits: 1000, // Initialize credits
          subscription_status: 'active', // Initialize status
          // Add other profile fields as needed, e.g., name, image
        },
      ]);

    if (insertError) {
      console.error('Error inserting user profile:', insertError);
      await supabaseAdmin.auth.admin.deleteUser(userId);
      return NextResponse.json({ error: 'Failed to create user profile.' }, { status: 500 });
    }

    // **Generate Confirmation Link**
    const { data: linkData, error: linkError } = await supabaseAdmin.auth.admin.generateLink({
      type: 'signup', // Type of link to generate
      email: email,    // User's email
      password: password, // User's password (optional for 'signup' type)
    });

    if (linkError) {
      console.error('Error generating confirmation link:', linkError);
      return NextResponse.json(
        { error: 'User registered, but failed to generate confirmation link.' },
        { status: 500 }
      );
    }

    if (!linkData?.properties?.action_link) {
      console.error('Confirmation link is undefined.');
      return NextResponse.json(
        { error: 'User registered, but confirmation link is undefined.' },
        { status: 500 }
      );
    }

    const confirmationLink = linkData.properties.action_link;

    // **Send Confirmation Email via SendGrid**
    const msg = {
      to: email, // Recipient's email
      from: process.env.EMAIL_FROM!, // Verified sender
      subject: 'Confirm Your Email Address',
      text: `Hello,

Please confirm your email address by clicking the link below:

${confirmationLink}

If you did not sign up for this account, please ignore this email.

Thank you!
`,
      html: `<p>Hello,</p>
<p>Please confirm your email address by clicking the link below:</p>
<a href="${confirmationLink}">Confirm Email</a>
<p>If you did not sign up for this account, please ignore this email.</p>
<p>Thank you!</p>`,
    };

    try {
      await sgMail.send(msg);
      console.log('Confirmation email sent to:', email);
    } catch (sendError) {
      console.error('Error sending confirmation email:', sendError);
      return NextResponse.json(
        { error: 'User registered, but failed to send confirmation email.' },
        { status: 500 }
      );
    }

    // If everything succeeds
    return NextResponse.json(
      { message: 'User registered successfully. Please check your email to confirm your address.' },
      { status: 201 }
    );
  } catch (error) {
    console.error('Error in /api/auth/register:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}