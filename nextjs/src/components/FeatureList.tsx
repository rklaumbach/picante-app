// src/components/FeatureList.tsx

import React from 'react';

interface FeatureListProps {
  features: string[];
}

const FeatureList: React.FC<FeatureListProps> = ({ features }) => {
  return (
    <ul className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-lg">
      {features.map((feature, index) => (
        <li key={index} className="flex items-center p-4 bg-red-600 rounded-lg shadow">
          <span className="text-white">{feature}</span>
        </li>
      ))}
    </ul>
  );
};

export default FeatureList;
