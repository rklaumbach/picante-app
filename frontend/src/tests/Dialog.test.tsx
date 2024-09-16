// src/tests/Dialog.test.tsx

import React from 'react';
import { render, screen } from '@testing-library/react';
import Dialog from '../components/Dialog';

test('Dialog displays children when isOpen is true', () => {
  render(
    <Dialog isOpen={true} onClose={() => {}}>
      <div data-testid="dialog-content">Dialog Content</div>
    </Dialog>
  );
  const content = screen.getByTestId('dialog-content');
  expect(content).toBeInTheDocument();
});

test('Dialog does not display when isOpen is false', () => {
  render(
    <Dialog isOpen={false} onClose={() => {}}>
      <div data-testid="dialog-content">Dialog Content</div>
    </Dialog>
  );
  const content = screen.queryByTestId('dialog-content');
  expect(content).not.toBeInTheDocument();
});
