const mongoose = require('mongoose');

const imageSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  imageUrl: { type: String, required: true },
  bodyPrompt: { type: String },
  facePrompt: { type: String },
  timestamp: { type: Date, default: Date.now },
  resolution: { type: String },
});

module.exports = mongoose.model('Image', imageSchema);
