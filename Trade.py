import streamlit as st
import torch
from diffusers import DiffusionPipeline
import time
import io
from huggingface_hub import HfApi

# Configure page
st.set_page_config(
    page_title="Pony SDXL Image Generator",
    page_icon="🐎",
    layout="wide"
)

# Initialize session states
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_pipeline():
    """Load and cache the model pipeline"""
    # Updated model ID to the correct one
    model_id = "SG161222/RealVisXL_V4.0"  # Using a stable SDXL model
    
    try:
        if torch.cuda.is_available():
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to("cuda")
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                use_safetensors=True
            ).to("cpu")
        
        # Enable memory efficient attention if possible
        if torch.cuda.is_available():
            pipeline.enable_xformers_memory_efficient_attention()
        
        return pipeline
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have a stable internet connection and sufficient disk space.")
        return None

def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height):
    """Generate image using the pipeline"""
    pipeline = load_pipeline()
    
    if pipeline is None:
        return None
    
    try:
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            )
            
        return result.images[0]
    
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None

def main():
    st.title("🐎 Pony SDXL Image Generator")
    st.markdown("Create beautiful pony-styled images using SDXL")

    # Sidebar for settings
    with st.sidebar:
        st.header("Generation Settings")
        
        width = st.select_slider(
            "Image Width",
            options=[512, 768, 1024],
            value=1024
        )
        
        height = st.select_slider(
            "Image Height",
            options=[512, 768, 1024],
            value=1024
        )
        
        steps = st.slider(
            "Generation Steps",
            min_value=20,
            max_value=100,
            value=30,
            help="More steps = better quality but slower generation"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            help="How closely to follow the prompt"
        )
        
        st.header("System Info")
        device = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.write(f"Using: {device}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Prompt",
            placeholder="Describe the image you want to generate...",
            help="Be specific and detailed in your description"
        )
        
        negative_prompt = st.text_area(
            "Negative Prompt",
            value="ugly, deformed, noisy, blurry, low quality, distorted, bad anatomy",
            help="Describe what you don't want in the image"
        )

    with col2:
        st.info("""
        **Tips for better results:**
        - Be specific in your descriptions
        - Include details about style and composition
        - Add "pony style, mlp style" to get pony-like results
        - Mention lighting and atmosphere
        - Use artistic terms for better results
        """)

    # Generate button
    if st.button("🎨 Generate Image", type="primary"):
        if not prompt:
            st.warning("Please enter a prompt first!")
            return
            
        try:
            with st.spinner("🎨 Creating your masterpiece..."):
                start_time = time.time()
                
                # Add pony-style keywords to prompt if not present
                if "pony" not in prompt.lower():
                    prompt += ", pony style, mlp style, cute pony"
                
                image = generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                )
                
                if image is None:
                    return
                
                generation_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    "prompt": prompt,
                    "image": image,
                    "time": generation_time,
                    "settings": {
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "size": f"{width}x{height}"
                    }
                })
            
            st.success(f"Image generated in {generation_time:.2f} seconds!")

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            return

    # Display history
    if st.session_state.history:
        st.header("Generated Images")
        
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(
                f"Image {len(st.session_state.history) - idx}",
                expanded=(idx == 0)
            ):
                st.image(item["image"], use_column_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Prompt:**", item["prompt"])
                    st.write("**Generation Time:**", f"{item['time']:.2f}s")
                
                with col2:
                    st.write("**Settings:**")
                    st.write(f"- Size: {item['settings']['size']}")
                    st.write(f"- Steps: {item['settings']['steps']}")
                    st.write(f"- Guidance Scale: {item['settings']['guidance_scale']}")
                
                # Save button
                img_bytes = io.BytesIO()
                item["image"].save(img_bytes, format="PNG")
                st.download_button(
                    "💾 Save Image",
                    img_bytes.getvalue(),
                    f"pony_generation_{int(time.time())}.png",
                    "image/png"
                )

if __name__ == "__main__":
    main()
