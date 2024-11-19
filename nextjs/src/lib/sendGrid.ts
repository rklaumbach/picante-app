// src/lib/sendGrid.ts

import sgMail from '@sendgrid/mail';

sgMail.setApiKey(process.env.EMAIL_SERVER_PASSWORD!);

export { sgMail };
