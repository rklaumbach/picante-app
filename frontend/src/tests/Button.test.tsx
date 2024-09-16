// src/tests/Button.test.tsx

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Button from '../components/Button';

test('renders Button component with text', () => {
  render(<Button text="Click Me" />);
  const buttonElement = screen.getByText(/Click Me/i);
  expect(buttonElement).toBeInTheDocument();
});

test('Button click triggers onClick handler', () => {
  const handleClick = jest.fn();
  render(<Button text="Click Me" onClick={handleClick} />);
  const buttonElement = screen.getByText(/Click Me/i);
  fireEvent.click(buttonElement);
  expect(handleClick).toHaveBeenCalledTimes(1);
});
