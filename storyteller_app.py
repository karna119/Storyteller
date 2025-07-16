#!/usr/bin/env python3
"""
GPT-2 Storyteller Application with Streamlit Interface
A web-based interactive story generator using GPT-2 model
"""

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="ByteXL- Story Teller",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StoryTeller:
    def __init__(self, model_name="gpt2"):
        """Initialize the storyteller with GPT-2 model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_story(self, prompt, max_length=200, temperature=0.8, top_p=0.9, top_k=50):
        """Generate a story based on the given prompt"""
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate story
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return only the generated part (remove the original prompt)
            story = generated_text[len(prompt):].strip()
            
            return f"{prompt}{story}"
            
        except Exception as e:
            return f"Error generating story: {str(e)}"

@st.cache_resource
def load_model():
    """Load the model with caching for better performance"""
    with st.spinner("Loading GPT-2 model... This may take a moment."):
        return StoryTeller()

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("📚 ByteXL Story Teller")
    st.markdown("""
    Welcome to the AI-powered story generator! Enter a prompt and watch as GPT-2 creates an engaging story for you.
    
    **How to use:**
    1. Enter your story prompt in the text area below
    2. Adjust the generation parameters in the sidebar
    3. Click "Generate Story" to create your story
    """)
    
    # Load the model
    storyteller = load_model()
    
    # Sidebar for parameters
    st.sidebar.header("🎛️ Generation Parameters")
    
    max_length = st.sidebar.slider(
        "Max Length",
        min_value=50,
        max_value=500,
        value=200,
        step=10,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Controls randomness (higher = more creative)"
    )
    
    top_p = st.sidebar.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling parameter"
    )
    
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Limits vocabulary for each step"
    )
    
    # Device info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(f"🖥️ Using device: {storyteller.device}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Story prompt input
        prompt = st.text_area(
            "Story Prompt",
            placeholder="Once upon a time in a magical forest...",
            height=100,
            help="Enter the beginning of your story here"
        )
        
        # Generate button
        if st.button("Generate Story 📖", type="primary", use_container_width=True):
            if prompt.strip():
                with st.spinner("Generating your story... Please wait."):
                    story = storyteller.generate_story(
                        prompt=prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
                    
                    # Store in session state
                    st.session_state.generated_story = story
            else:
                st.error("Please enter a story prompt!")
        
        # Display generated story
        if hasattr(st.session_state, 'generated_story'):
            st.markdown("### 📖 Generated Story")
            st.text_area(
                "Your Story",
                value=st.session_state.generated_story,
                height=300,
                help="Your generated story appears here"
            )
            
            # Copy button (using markdown since Streamlit doesn't have native copy)
            st.markdown("💡 **Tip:** Select all text above and copy it to save your story!")
    
    with col2:
        # Example prompts
        st.markdown("### 💡 Example Prompts")
        
        example_prompts = [
            "Once upon a time in a magical forest, there lived a young wizard who",
            "In the year 2150, humanity discovered a mysterious signal from deep space",
            "The old lighthouse keeper had been alone for thirty years when suddenly",
            "Detective Sarah Chen walked into the abandoned mansion and noticed",
            "The last dragon on Earth was hiding in the mountains when"
        ]
        
        for i, example in enumerate(example_prompts):
            if st.button(f"📝 Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_prompt = example
                st.rerun()
        
        # If an example was selected, update the prompt
        if hasattr(st.session_state, 'example_prompt'):
            st.info(f"Example selected: {st.session_state.example_prompt[:50]}...")
        
        # Tips section
        st.markdown("### 📝 Tips for Better Stories")
        st.markdown("""
        - Start with an engaging opening line
        - Include specific details (names, places, emotions)
        - Try different temperature settings for varied creativity
        - Experiment with different prompt styles
        - Lower temperature = more focused
        - Higher temperature = more creative
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using NIMBUS and GEMINIAI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

