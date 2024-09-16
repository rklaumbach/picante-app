// src/components/DeleteConfirmationDialog.tsx

import React from 'react';
import Dialog from './Dialog';
import Button from './Button';

interface DeleteConfirmationDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

const DeleteConfirmationDialog: React.FC<DeleteConfirmationDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
}) => {
  return (
    <Dialog isOpen={isOpen} onClose={onClose}>
      <h2 className="text-2xl mb-4">Permanently Delete Image?</h2>
      <p className="mb-6">It will be lost forever. Are you sure?</p>
      <div className="flex justify-end">
        <Button text="No" className="bg-gray-300 text-black mr-2" onClick={onClose} />
        <Button text="Yes" className="bg-red-500 text-white" onClick={onConfirm} />
      </div>
    </Dialog>
  );
};

export default DeleteConfirmationDialog;
