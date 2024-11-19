// src/app/layout.tsx

import Providers from './providers';
import './globals.css';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css'; // Import CSS

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head />
      <body>
        <Providers>
          {children}
          <ToastContainer position="top-center" autoClose={5000} />
        </Providers>
      </body>
    </html>
  );
}
