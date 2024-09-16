// frontend/src/tests/DeleteConfirmationDialog.test.tsx

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import DeleteConfirmationDialog from '../components/DeleteConfirmationDialog';

test('DeleteConfirmationDialog renders and functions correctly', () => {
  const handleClose = jest.fn();
  const handleConfirm = jest.fn();
  
  render(<DeleteConfirmationDialog isOpen={true} onClose={handleClose} onConfirm={handleConfirm} />);
  
  expect(screen.getByText(/Permanently Delete Image\?/i)).toBeInTheDocument();
  expect(screen.getByText(/It will be lost forever. Are you sure\?/i)).toBeInTheDocument();

  fireEvent.click(screen.getByText(/No/i));
  expect(handleClose).toHaveBeenCalled();

  fireEvent.click(screen.getByText(/Yes/i));
  expect(handleConfirm).toHaveBeenCalled();
});
