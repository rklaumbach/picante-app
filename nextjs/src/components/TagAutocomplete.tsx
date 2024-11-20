// src/app/components/TagAutocomplete.tsx

'use client';

import { Combobox, Label, ComboboxInput, ComboboxButton, ComboboxOptions, ComboboxOption } from '@headlessui/react';
import React, { useState, useEffect, Fragment, useRef } from 'react';

// Define the structure for each selected tag with its weight
interface SelectedTag {
  tag: string;
  weight: number;
}

interface Tag {
  tag: string;
  post_count: number;
}

interface TagAutocompleteProps {
  label: string;
  selectedTags: SelectedTag[];
  setSelectedTags: (tags: SelectedTag[]) => void;
}

const TagAutocomplete: React.FC<TagAutocompleteProps> = ({ label, selectedTags, setSelectedTags }) => {
  const [allTags, setAllTags] = useState<Tag[]>([]);
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch tags from tags.json on component mount
  useEffect(() => {
    const fetchTags = async () => {
      try {
        const response = await fetch('/tags.json');
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const tagsData: Tag[] = await response.json();
        setAllTags(tagsData);
      } catch (error) {
        console.error('Failed to fetch tags:', error);
      }
    };

    fetchTags();
  }, []);

  // Filter tags based on the query
  const filteredTags =
    query === ''
      ? allTags
          .sort((a, b) => b.post_count - a.post_count)
          .slice(0, 10)
          .map((tag) => tag.tag)
      : allTags
          .filter((tagObj) => tagObj.tag.toLowerCase().includes(query.toLowerCase()))
          .sort((a, b) => b.post_count - a.post_count)
          .slice(0, 10)
          .map((tagObj) => tagObj.tag);

  // Handle tag selection (selection from Combobox or free solo input)
  const handleSelection = (tag: string | null) => {
    if (tag) {
      // Prevent duplicate tags
      if (!selectedTags.some((t) => t.tag.toLowerCase() === tag.toLowerCase())) {
        setSelectedTags([...selectedTags, { tag, weight: 1.0 }]);
      }
      setQuery(''); // Clear the input field

      // Capture the current input element
      const currentInput = inputRef.current;

      // Focus the input field after adding the tag
      if (currentInput) {
        setTimeout(() => {
          currentInput.value = ''; // Directly set the input value to empty
          currentInput.focus(); // Refocus the input for the next entry
        }, 0);
      }
    }
  };
    

  // Increase the weight of a selected tag
  const handleIncreaseWeight = (index: number) => {
    const updatedTags = [...selectedTags];
    const newWeight = parseFloat((updatedTags[index].weight + 0.1).toFixed(1));
    if (newWeight <= 1.5) {
      updatedTags[index].weight = newWeight;
      setSelectedTags(updatedTags);
    }
  };

  // Decrease the weight of a selected tag
  const handleDecreaseWeight = (index: number) => {
    const updatedTags = [...selectedTags];
    const newWeight = parseFloat((updatedTags[index].weight - 0.1).toFixed(1));
    if (newWeight >= 0.5) {
      updatedTags[index].weight = newWeight;
      setSelectedTags(updatedTags);
    }
  };

  // Remove a selected tag
  const handleRemoveTag = (index: number) => {
    const updatedTags = [...selectedTags];
    updatedTags.splice(index, 1);
    setSelectedTags(updatedTags);
  };

  return (
    <div className="w-full mb-6">
      <label className="block text-xl font-bold text-gray-700 mb-3">{label}:</label>
      {/* Display Selected Tags as Removable Chips */}
      <div className="flex flex-wrap mt-6 gap-4">
        {selectedTags.map((selectedTag, index) => (
          <div
            key={`${selectedTag.tag}-${index}`}
            className="flex items-center bg-blue-200 text-blue-900 px-3 py-1 rounded-full text-2xl"
          >
            {/* Decrease Weight Button */}
            <button
              onClick={() => handleDecreaseWeight(index)}
              className="text-blue-700 hover:text-blue-900 focus:outline-none p-2"
              aria-label={`Decrease weight of ${selectedTag.tag}`}
            >
              &minus;
            </button>

            {/* Increase Weight Button */}
            <button
              onClick={() => handleIncreaseWeight(index)}
              className="text-blue-700 hover:text-blue-900 focus:outline-none p-2"
              aria-label={`Increase weight of ${selectedTag.tag}`}
            >
              +
            </button>

            {/* Tag Label with Weight */}
            <span className="mx-8 text-xl">
              {selectedTag.tag} ({selectedTag.weight.toFixed(1)})
            </span>

            {/* Remove Tag Button */}
            <button
              onClick={() => handleRemoveTag(index)}
              className="text-red-700 hover:text-red-900 focus:outline-none p-4"
              aria-label={`Remove ${selectedTag.tag}`}
            >
              &times;
            </button>
          </div>
        ))}
      </div>

      {/* Combobox for Tag Autocomplete */}
      <Combobox onChange={handleSelection}>
        <div className="relative">
          <ComboboxInput
            ref={inputRef}
            className="w-full border border-gray-400 rounded-md shadow-sm pl-4 pr-10 py-4 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-xl"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Start typing to add tags"
            displayValue={(tag: string) => tag}
          />
          <ComboboxButton className="absolute inset-y-0 right-0 flex items-center pr-2">
            {/* Optional: Add an icon here if desired */}
            {/* Example: ChevronDownIcon */}
          </ComboboxButton>

          {filteredTags.length > 0 && (
            <ComboboxOptions className="absolute z-10 mt-2 w-full bg-white shadow-lg max-h-72 rounded-md py-2 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none text-xl">
              {filteredTags.map((tag) => (
                <ComboboxOption key={tag} value={tag} as={Fragment}>
                  {({ active }) => (
                    <li
                      className={`cursor-pointer select-none relative py-3 pl-5 pr-5 list-none ${
                        active ? 'bg-blue-600 text-white' : 'text-gray-900'
                      }`}
                    >
                      <span className="block truncate">{tag}</span>
                    </li>
                  )}
                </ComboboxOption>
              ))}
            </ComboboxOptions>
          )}
        </div>
      </Combobox>
    </div>
  );
};

export default TagAutocomplete;
