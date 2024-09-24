# Create the top-level files
touch config.json requirements.txt main_notebook.ipynb

# Create the 'models' directory and its subdirectories
mkdir -p models/sdxxxl_v30
mkdir -p models/vae_model
mkdir -p models/clip_nat5xxl_fp16
mkdir -p models/clip_name2clip_1
mkdir -p models/controlnet_1
mkdir -p models/controlnet_2
mkdir -p models/controlnet_3

# Create the RealESRGAN model file (empty placeholder)
touch models/RealESRGAN_x2.pth

# Create the 'control_images' directory and its placeholder images
mkdir -p control_images
touch control_images/control_image_1.png control_images/control_image_2.png control_images/control_image_3.png

# Optional: Display the created directory structure
echo "Directory structure created successfully:"
tree comfyui_to_notebook

