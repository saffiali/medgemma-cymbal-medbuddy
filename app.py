import streamlit as st
import os
import base64
import logging
from io import BytesIO
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

# Attempt to import Google Cloud libraries. This provides a clear error if the required
# package is not installed, which is better than a generic ImportError later on.
try:
    from google.cloud import aiplatform
    from google.api_core import exceptions as core_exceptions
except ImportError:
    st.error("Google Cloud AI Platform library not found. Please install it using `pip install google-cloud-aiplatform`")
    st.stop()


# --- 1. Page Configuration & Logging ---
# Purpose: To set up the basic properties of the Streamlit page and configure logging
# for easier debugging. `st.set_page_config` must be the first Streamlit command.

st.set_page_config(
    page_title="MedGemma Image Analyser Demo",
    page_icon="images/app/gemini_avatar.png", # Make sure this path is correct
    layout="wide",
    initial_sidebar_state="auto",
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- 2. Model Initialization ---
# Purpose: To establish a connection to the Vertex AI model endpoint.
# The `@st.cache_resource` decorator is crucial here. It tells Streamlit to run this
# function only once and then cache the returned object (the endpoint connection).
# This prevents the app from re-establishing the connection every time the user interacts
# with the UI, making the app much faster and more efficient.

@st.cache_resource
def fn_initialize_model():
    """
    Initializes and caches the Vertex AI model endpoint resource.
    This prevents re-initializing the connection on every user interaction.
    """
    load_dotenv() # Loads details from a .env file
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_region = os.getenv("GCP_REGION")
    model_endpoint_id = os.getenv("MODEL_ENDPOINT_ID")
    model_endpoint_region = os.getenv("MODEL_ENDPOINT_REGION")

     # This validation ensures the app fails gracefully with a clear message if the .env file is missing or incomplete.
    if not all([gcp_project_id, gcp_region, model_endpoint_id, model_endpoint_region]):
        st.error("Missing Google Cloud configuration. Please check your .env file or environment variables.")
        st.stop()

    logger.info("Initializing Vertex AI API...")
    aiplatform.init(project=gcp_project_id, location=gcp_region)

    logger.info(f"Loading model endpoint: {model_endpoint_id}")
    try:
        endpoint = aiplatform.Endpoint(
            endpoint_name=model_endpoint_id,
            project=gcp_project_id,
            location=model_endpoint_region,
        )
        return endpoint
    except Exception as e:
        st.error(f"Failed to initialize AI Platform endpoint: {e}")
        logger.error(f"Endpoint initialization failed: {e}")
        st.stop()


# --- 3. Backend AI Query Function ---
# Purpose: To handle the logic of sending a request to the AI model.
# This function constructs the prompt, packages the user's text and the uploaded image
# into the format expected by the Vertex AI API, and handles the API call itself.

def fn_run_query(endpoint, input_text, max_tokens=500, temperature=0.0):
    """
    Sends a query to the Vertex AI model with both text and an image.

    Args:
        endpoint: The initialized Vertex AI endpoint.
        input_text (str): The user's prompt.
        max_tokens (int): Maximum number of tokens for the model to generate.
        temperature (float): The model's temperature (creativity).

    Returns:
        str: The model's prediction text or a formatted error message.
    """
    # A detailed system prompt guides the model to behave in a specific way, ensuring
    # consistent and high-quality responses.
    
   
    # system_instruction = """
    #     **Role and Goal:** You are an expert radiological assistant AI. Your primary goal is to provide a structured, accurate, and clinically relevant analysis of the provided medical image to assist a human radiologist.
    #     **Critical Safety Rule:** You MUST NOT invent, guess, or hallucinate findings that are not clearly visible in the image. Your analysis must be based solely on the visual data provided. If the image quality is insufficient for a confident assessment, you must state this as your primary finding.
    #     **Analysis Instructions:**
    #     1.  **Identify Anatomy:** Begin by identifying the body part, view, and any visible key anatomical structures.
    #     2.  **Describe Findings:** Systematically describe all observations, both normal and abnormal. For abnormalities, specify their location, size, shape, and characteristics (e.g., density, margins).
    #     3.  **Mention Pertinent Negatives:** Explicitly state the absence of significant expected abnormalities (e.g., "No evidence of acute fracture," "The lungs are clear with no infiltrates.").

    #     **Output Format:** Your response MUST be structured into the following two sections. Do not use conversational language.

    #     **FINDINGS:**
    #     - (Provide a detailed, objective, bulleted list of observations from your analysis here.)
    #     - (Describe normal and abnormal findings.)

    #     **IMPRESSION:**
    #     - (Provide a concise, numbered list summarizing the most critical findings and their likely clinical significance.)
    #     - (This is your conclusion. State the most likely diagnosis if confidence is high, or list differential diagnoses if uncertain.)
    #     - (If applicable, suggest potential next steps or correlations, e.g., "Clinical correlation is recommended.")
    # """
    
    system_instruction = """
        You are an expert in medical imaging metadata extraction. You work in R&D for a global life science company. 
        
        Your task is to analyze medical images across various specialities and formats (DICOM, Large Slides, TIFF) and generate all possible metadata tags for a given image in both JSON and DICOM formats.
            1. Analyze the provided medical image data.
            2. Extract ALL possible metadata tags applicable to the image.
            3. Generate the metadata in TWO formats: JSON and DICOM.
            4. Ensure that the JSON format includes ALL extracted metadata tags with their corresponding values.
            5. Ensure that the DICOM format adheres to the DICOM standard, including appropriate tag numbers and value representations.
            6. Output BOTH the JSON and DICOM formats.
            7. If you cannot complete your task with the information found in the medical image data, please respond with, "I'm sorry. My knowledge base is insufficient to complete this task." and then explain what information is needed for your knowledge base to complete the task.
    """

    # Set user prompt
    prompt = input_text
    #formatted_prompt = f"{system_instruction} {prompt} <start_of_image>"
    img_b64 = st.session_state.img_b64
    data_url = f"data:image/png;base64,{img_b64}"

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                        ]
        }
    ]

    # The 'instances' list is the payload sent to the Vertex AI endpoint.
    instances = [
        {
            "@requestFormat": "chatCompletions",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
    ]

    # The try...except block gracefully handles potential issues during the API call,
    # such as network errors or invalid model responses.
    try:
        logger.info("Sending prediction request to Vertex AI...")
        response = endpoint.predict(instances=instances, use_dedicated_endpoint=True)

        # Safely access the prediction to avoid KeyErrors if the response format is unexpected.
        # This is more robust than direct key access.
        prediction = response.predictions["choices"][0]["message"]["content"]

        if not prediction:
            logger.warning(f"Model returned an empty or malformed response: {response.predictions[0]}")
            return "The model returned an empty response. This could be due to a content safety filter or an internal model issue."

        return prediction

    # Catch specific, common API errors for more targeted feedback.
    except core_exceptions.PermissionDenied as e:
        logger.error(f"Permission Denied calling Vertex AI: {e}", exc_info=True)
        return "Error: Permission denied. Please check the application's service account permissions for the Vertex AI Endpoint."
    except core_exceptions.ResourceExhausted as e:
        logger.error(f"Resource Exhausted calling Vertex AI: {e}", exc_info=True)
        return "Error: The service is currently busy or has exceeded its quota. Please try again in a few moments."
    except core_exceptions.InvalidArgument as e:
        logger.error(f"Invalid Argument sent to Vertex AI: {e}", exc_info=True)
        return "Error: The request sent to the model was invalid. This might be due to a malformed image or prompt."
    # A general catch-all for other Google Cloud API errors.
    except core_exceptions.GoogleAPICallError as e:
        logger.error(f"A Google Cloud API call error occurred: {e}", exc_info=True)
        return f"An API error occurred. Please check the application logs for details."
    except Exception as e:
        # Catch any other unexpected exceptions. exc_info=True includes the traceback in the log.
        logger.error(f"An unexpected exception occurred during prediction: {e}", exc_info=True)
        return f"An unexpected error occurred. Please check the logs for details."


# --- 4. UI Rendering Functions ---
# Purpose: To organize the code responsible for drawing the user interface.
# Breaking the UI into functions makes the main app logic cleaner and easier to read.

def fn_format_chat_history():
    """
    Formats the entire chat history into a single string for text file export.

    Returns:
        str: A formatted string containing the chat history.
    """
    history_str = "Cymbal MedBuddy Chat History\n"
    history_str += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    history_str += "="*40 + "\n\n"

    for message in st.session_state.messages:
        # Use a more descriptive role name for the export file.
        role = "User" if message["role"] == "user" else "MedBuddy"
        history_str += f"**{role}:**\n{message['content']}\n\n"
        history_str += "-"*40 + "\n\n"

    return history_str


def fn_render_sidebar():
    """
    Renders all elements in the Streamlit sidebar.
    """
    with st.sidebar:
        # Make sure the image path is correct or comment out if not needed
        st.image("images/app/gemini_avatar.png", width=100)
        st.markdown("<h1 style='text-align: center;'>Cymbal MedBuddy</h1>", unsafe_allow_html=True)
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload a medical scan", type=["jpg", "jpeg", "png"], key="file_uploader"
        )
        if uploaded_file:
            try:
                # Process the uploaded image and store its base64 representation in the session state.
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Scan", use_container_width=True)

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                st.session_state.img_b64 = img_b64
            except Exception as e:
                st.error(f"Error processing image file: {e}")

        st.markdown("---")
        if st.button("Clear Chat History"):
            # Resets the session state to its initial values.
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! Please upload a medical scan to get started."}
            ]
            if "img_b64" in st.session_state:
                del st.session_state.img_b64
            # Use st.rerun() to immediately apply the changes
            st.rerun() # Reruns the script from the top to ensure a clean UI reset.

        # Add download button for chat history. It only appears if there's a conversation to save.
        if len(st.session_state.get("messages", [])) > 1:
            chat_export_data = fn_format_chat_history()
            st.download_button(
                label="📥 Export Chat History",
                data=chat_export_data,
                file_name=f"medbuddy_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download the current chat conversation as a text file."
            )

        st.markdown("---")
        # Use an expander to keep the UI clean. It holds advanced settings.
        with st.expander("Model Parameters", expanded=False):
            st.slider(
                label="Temperature",
                min_value=0.0,
                max_value=1.0,
                key="temperature", # The key automatically links this to st.session_state.temperature
                step=0.05,
                help="Controls randomness. Lower values make the model more deterministic and focused, while higher values make it more creative."
            )
            st.slider(
                label="Max Output Tokens",
                min_value=256,
                max_value=500,
                key="max_tokens", # The key automatically links this to st.session_state.max_tokens
                step=64,
                help="Sets the maximum number of tokens (words/sub-words) the model can generate in a single response."
            )

def fn_render_chat_interface(avatars):
    """
    Renders the main chat message history.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=avatars.get(message["role"])):
            st.markdown(message["content"])

def fn_handle_user_input(endpoint, avatars):
    """
    Handles user chat input, triggers model query, and displays the response.
    Disables the chat input if no image is uploaded.
    """
    # This logic is key to good UX: the chat input is disabled, and a helpful message is shown
    # until the user uploads an image. `st.session_state.get()` is used to safely check for the key.
    is_disabled = not st.session_state.get("img_b64")
    
    # Provide a clear message to the user next to the chat input
    if is_disabled:
        st.info("Please upload an image in the sidebar to enable the chat.")

    prompt = st.chat_input("Ask a question about the scan...", disabled=is_disabled)

    if prompt:
        # Appends the user message to state and displays it immediately.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=avatars["user"]):
            st.markdown(prompt)

        # Displays a spinner while waiting for the model's response.
        with st.chat_message("assistant", avatar=avatars["assistant"]):
            with st.spinner("Analyzing the scan..."):
                response = fn_run_query(
                    endpoint=endpoint,
                    input_text=prompt,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )
                st.markdown(response)
        
        # Appends the assistant's response to the session state.
        st.session_state.messages.append({"role": "assistant", "content": response})


# --- 5. Main Application Logic ---
# Purpose: To orchestrate the entire application. It calls the other functions in the correct order.

def main():
    """
    Main function to orchestrate the Streamlit app.
    """
    logger.info("Starting Cymbal MedBuddy application...")
    
    # Define avatar paths. Ensure these files exist in the specified locations.
    avatars = {"assistant": "images/app/gemini_avatar.png", "user": "images/app/user_avatar.png"}

    # Initialize the model once and cache it
    endpoint = fn_initialize_model()

    # Initialize session state only if it doesn't already exist. This ensures that the
    # chat history persists across reruns (e.g., when a user uploads a new image).
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Please upload a medical scan to get started."}
        ]

    # Initialize model parameters in session state if they don't exist.
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1  # A sensible default for clinical accuracy
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 500 # A reasonable default length

    # Render UI components in order
    fn_render_sidebar()
    st.title("AI-Powered Medical Image Analysis")
    st.markdown("Your assistant for interpreting medical scans. Upload a scan via the sidebar and ask a question to begin.")
    st.markdown("---")
    
    fn_render_chat_interface(avatars)
    fn_handle_user_input(endpoint, avatars)

# --- 6. Execution Guard ---
# Purpose: This is a standard Python convention. It ensures that the `main()` function is called
# only when the script is executed directly (not when it's imported as a module by another script).
if __name__ == "__main__":
    main()
