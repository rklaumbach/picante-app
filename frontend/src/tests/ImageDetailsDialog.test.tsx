// frontend/src/tests/ImageDetailsDialog.test.tsx

import React from 'react';
import { render, screen } from '@testing-library/react';
import ImageDetailsDialog from '../components/ImageDetailsDialog';

test('ImageDetailsDialog displays image details', () => {
  const imageDetails = {
    timestamp: '2022-10-01T12:00:00Z',
    resolution: '1024x768',
    bodyPrompt: 'Test body prompt',
    facePrompt: 'Test face prompt',
  };

  render(<ImageDetailsDialog isOpen={true} onClose={() => {}} imageDetails={imageDetails} />);

  expect(screen.getByText(/Timestamp:/i)).toHaveTextContent('Timestamp: 2022-10-01T12:00:00Z');
  expect(screen.getByText(/Resolution:/i)).toHaveTextContent('Resolution: 1024x768');
  expect(screen.getByText(/Body Prompt:/i)).toHaveTextContent('Body Prompt: Test body prompt');
  expect(screen.getByText(/Face Prompt:/i)).toHaveTextContent('Face Prompt: Test face prompt');
});
