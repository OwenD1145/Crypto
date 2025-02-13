import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from io import BytesIO

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
            # variant="fp16"
        ).to("cuda")
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            use_safetensors=True,
        ).to("cpu")
    
    return pipeline

def create_pony_prompt(base_prompt):
    """Enhance prompt with pony-specific styling"""
    style_prompt = ("masterpiece, best quality, highly detailed, "
                   "beautiful lighting, vibrant colors")
    return f"{base_prompt}, {style_prompt}"

def generate_image(pipeline, prompt, negative_prompt="", guidance_scale=7.5, steps=30):
    """Generate image using the pipeline"""
    with torch.inference_mode():
        image = pipeline(
            prompt=create_pony_prompt(prompt),
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        ).images[0]
    return image

def main():
    st.set_page_config(
        page_title="Pony XL Image Generator",
        page_icon="🐎",
        layout="wide"
    )
    
    st.title("🐎 Pony XL Image Generator")
    st.markdown("Create beautiful pony-style images using AI")
    
    # Display device info
    device_info = "🖥️ Using GPU" if torch.cuda.is_available() else "🖥️ Using CPU (generation will be slower)"
    st.info(device_info)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Prompt input
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="Example: A cute pony in a magical forest",
            height=100
        )
        
        # Negative prompt input
        negative_prompt = st.text_area(
            "Negative prompt (what to avoid):",
            placeholder="Example: blurry, bad quality, distorted",
            height=100
        )
    
    with col2:
        # Advanced settings
        st.subheader("Generation Settings")
        guidance_scale = st.slider(
            "Guidance Scale (how closely to follow the prompt)",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5
        )
        
        steps = st.slider(
            "Number of Steps",
            min_value=20,
            max_value=100,
            value=30,
            step=1
        )

    # Generate button
    if st.button("🎨 Generate Image", use_container_width=True):
        if prompt:
            try:
                with st.spinner("🔄 Loading model... (this might take a few minutes the first time)"):
                    pipeline = load_model()
                
                progress_text = "🎨 Creating your masterpiece..."
                with st.spinner(progress_text):
                    image = generate_image(
                        pipeline,
                        prompt,
                        negative_prompt,
                        guidance_scale,
                        steps
                    )
                
                # Display the generated image
                st.image(
                    image,
                    caption=f"Generated Image: {prompt}",
                    use_column_width=True
                )
                
                # Create download button
                buf = BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Image",
                    data=byte_im,
                    file_name="pony_xl_generated.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                if not torch.cuda.is_available():
                    st.warning("⚠️ Running on CPU mode. This model works best with a GPU.")
                st.info("💡 Tip: Try reducing the image quality settings if you're running out of memory.")
        else:
            st.warning("⚠️ Please enter a prompt first!")
    
    # Add helpful information at the bottom
    with st.expander("ℹ️ Tips for better results"):
        st.markdown("""
        - Be specific in your descriptions
        - Include details about lighting, atmosphere, and style
        - Use the negative prompt to remove unwanted elements
        - Adjust the guidance scale: higher values = closer to prompt but less creative
        - More steps generally mean better quality but slower generation
        """)

if __name__ == "__main__":
    main()
