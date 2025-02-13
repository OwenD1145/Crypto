import streamlit as st
import torch
from diffusers import DiffusionPipeline
import time
import io

# Verify GPU availability immediately
if not torch.cuda.is_available():
    st.error("❌ This application requires a CUDA-enabled GPU to run.")
    st.info("Please make sure you have:\n1. An NVIDIA GPU\n2. CUDA drivers installed\n3. PyTorch with CUDA support")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Pony SDXL Image Generator (GPU)",
    page_icon="🐎",
    layout="wide"
)

# Initialize session states
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_pipeline():
    """Load and cache the model pipeline"""
    model_id = "SG161222/RealVisXL_V4.0"
    
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")
        
        # Enable memory efficient attention
        pipeline.enable_xformers_memory_efficient_attention()
        
        # Enable memory optimization
        pipeline.enable_model_cpu_offload()
        
        return pipeline
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please check your GPU configuration and available memory.")
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
            
        # Clear GPU memory
        torch.cuda.empty_cache()
            
        return result.images[0]
    
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None

def main():
    st.title("🐎 Pony SDXL Image Generator (GPU Mode)")
    st.markdown("Create beautiful pony-styled images using SDXL")

    # Display GPU info
    gpu_info = f"🚀 Using GPU: {torch.cuda.get_device_name(0)}"
    st.success(gpu_info)

    # Sidebar for settings
    with st.sidebar:
        st.header("Generation Settings")
        
        width = st.select_slider(
            "Image Width",
            options=[512, 768, 1024],
            value=768,
            help="Larger sizes require more GPU memory"
        )
        
        height = st.select_slider(
            "Image Height",
            options=[512, 768, 1024],
            value=768,
            help="Larger sizes require more GPU memory"
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
        
        st.header("GPU Info")
        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
        st.write(f"CUDA Version: {torch.version.cuda}")
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        st.write(f"Memory Allocated: {memory_allocated:.2f} GB")
        st.write(f"Memory Reserved: {memory_reserved:.2f} GB")

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
        
        **GPU Mode Features:**
        - Faster generation times
        - Higher resolution support
        - Better memory management
        - FP16 precision for efficiency
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
            torch.cuda.empty_cache()  # Clear GPU memory on error
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
