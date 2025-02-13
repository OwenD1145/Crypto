import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import io
import time

# Page config
st.set_page_config(
    page_title="Pony XL Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state for generated images
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

@st.cache_resource
def load_model():
    """Load the Pony XL model"""
    model_id = "John6666/real-hybrid-pony-xl-v10-sdxl"
    
    # Check if CUDA is available and set dtype accordingly
    if torch.cuda.is_available():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            use_safetensors=True,
        ).to("cpu")
    
    return pipeline

def generate_image(prompt, negative_prompt, num_steps, guidance_scale):
    """Generate image from prompt"""
    pipeline = load_model()
    
    with torch.inference_mode():
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    
    return image

def main():
    st.title("🎨 Pony XL Image Generator")
    st.write("Generate unique images using the real-hybrid-pony-xl-v10-sdxl model")

    # Sidebar controls
    with st.sidebar:
        st.header("Generation Settings")
        num_steps = st.slider("Number of Steps", 20, 100, 30)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
        
        st.header("System Info")
        device = "GPU 🚀" if torch.cuda.is_available() else "CPU 💻"
        st.write(f"Using device: {device}")
        
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt",
            height=100,
            placeholder="Describe the image you want to generate..."
        )
        
        negative_prompt = st.text_area(
            "Enter negative prompt (what you don't want in the image)",
            height=100,
            placeholder="Elements you want to avoid in the generation...",
            value="ugly, deformed, noisy, blurry, low quality, distorted"
        )

    with col2:
        st.info("""
        💡 **Tips for better results:**
        - Be specific in your descriptions
        - Include details about style, lighting, and composition
        - Use the negative prompt to avoid unwanted elements
        """)

    # Generate button
    if st.button("🎨 Generate Image"):
        if not prompt:
            st.warning("Please enter a prompt first!")
            return
        
        try:
            with st.spinner("🎨 Generating your image... Please wait..."):
                start_time = time.time()
                
                # Generate the image
                generated_image = generate_image(
                    prompt,
                    negative_prompt,
                    num_steps,
                    guidance_scale
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Add to session state
                st.session_state.generated_images.append({
                    'image': generated_image,
                    'prompt': prompt,
                    'time': generation_time
                })
            
            st.success(f"Image generated in {generation_time:.2f} seconds!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return

    # Display generated images
    if st.session_state.generated_images:
        st.header("Generated Images")
        for idx, item in enumerate(reversed(st.session_state.generated_images)):
            with st.expander(f"Image {len(st.session_state.generated_images) - idx}", expanded=(idx == 0)):
                st.image(item['image'], use_column_width=True)
                st.write(f"**Prompt:** {item['prompt']}")
                st.write(f"**Generation Time:** {item['time']:.2f} seconds")
                
                # Add download button
                buf = io.BytesIO()
                item['image'].save(buf, format='PNG')
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name=f"generated_image_{idx}.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
