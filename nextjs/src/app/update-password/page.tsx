// src/app/update-password/page.tsx

'use client';

import React, { Suspense } from 'react';
import Header from '../../components/Header';
import PasswordUpdateForm from '../../components/PasswordUpdateForm';

const UpdatePasswordPage: React.FC = () => {
  return (
    <>
      <Header title="Update Password" />
      <Suspense fallback={<div>Loading...</div>}>
        <PasswordUpdateForm />
      </Suspense>
    </>
  );
};

export default UpdatePasswordPage;
