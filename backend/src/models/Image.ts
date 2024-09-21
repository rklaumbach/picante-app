import mongoose, { Document, Schema } from 'mongoose';

export interface IImage extends Document {
  userId: mongoose.Types.ObjectId;
  imageUrl: string;
  bodyPrompt?: string;
  facePrompt?: string;
  timestamp: Date;
  resolution?: string;
}

const imageSchema = new Schema<IImage>({
  userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
  imageUrl: { type: String, required: true },
  bodyPrompt: { type: String },
  facePrompt: { type: String },
  timestamp: { type: Date, default: Date.now },
  resolution: { type: String },
});

const Image = mongoose.model<IImage>('Image', imageSchema);

export default Image;
