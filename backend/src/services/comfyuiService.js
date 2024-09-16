const axios = require('axios');

const COMFYUI_API_BASE_URL = process.env.COMFYUI_API_BASE_URL;
const COMFYUI_API_KEY = process.env.COMFYUI_API_KEY;

exports.generateImage = async (prompts) => {
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
  } catch (error) {
    console.error('ComfyUI API error:', error.response ? error.response.data : error.message);
    throw new Error('ComfyUI API error');
  }
};
