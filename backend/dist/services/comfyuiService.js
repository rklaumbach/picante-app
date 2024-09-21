"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const axios_1 = __importDefault(require("axios"));
const COMFYUI_API_BASE_URL = process.env.COMFYUI_API_BASE_URL;
const COMFYUI_API_KEY = process.env.COMFYUI_API_KEY;
const generateImage = async (prompts) => {
    try {
        const response = await axios_1.default.post(`${COMFYUI_API_BASE_URL}/generate`, prompts, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${COMFYUI_API_KEY}`,
            },
        });
        // Assuming the API returns an object with an imageUrl property
        return response.data;
    }
    catch (error) {
        console.error('ComfyUI API error:', error.response ? error.response.data : error.message);
        throw new Error('ComfyUI API error');
    }
};
exports.default = { generateImage };
//# sourceMappingURL=comfyuiService.js.map