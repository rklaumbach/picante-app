import axios from 'axios';

const COMFYUI_API_BASE_URL = process.env.COMFYUI_API_BASE_URL!;
const COMFYUI_API_KEY = process.env.COMFYUI_API_KEY!;

interface GenerateImageResult {
  imageUrl: string;
  resolution?: string;
}

interface Prompts {
  bodyPrompt?: string;
  facePrompt?: string;
}

const generateImage = async (prompts: Prompts): Promise<GenerateImageResult> => {
  try {
    const response = await axios.post(
      `${COMFYUI_API_BASE_URL}/generate`,
      prompts,
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${COMFYUI_API_KEY}`,
        },
      }
    );

    // Assuming the API returns an object with an imageUrl property
    return response.data;
  } catch (error: any) {
    console.error('ComfyUI API error:', error.response ? error.response.data : error.message);
    throw new Error('ComfyUI API error');
  }
};

export default { generateImage };
