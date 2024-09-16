// frontend/src/tests/OutOfCreditsDialog.test.tsx

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import OutOfCreditsDialog from '../components/OutOfCreditsDialog';

test('renders OutOfCreditsDialog correctly', () => {
  const handleClose = jest.fn();
  render(<OutOfCreditsDialog isOpen={true} onClose={handleClose} />);
  
  expect(screen.getByText(/Out of Free Image Credits!/i)).toBeInTheDocument();
  expect(screen.getByText(/Free users of Picante can only generate up to 5 images per day/i)).toBeInTheDocument();

  const subscribeButton = screen.getByText(/Subscribe Now/i);
  fireEvent.click(subscribeButton);
  // Add expectations for navigation or actions performed
});
