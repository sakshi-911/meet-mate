import streamlit as st
import assemblyai as aai
from dotenv import load_dotenv
import os
import tempfile
import google.generativeai as genai

# Load environment variables
load_dotenv(override=True)
aai.settings.api_key = os.getenv('ASS_API_KEY')
genai.configure(api_key=os.getenv("GEN_AI_KEY"))

# App title and header
st.set_page_config(page_title="MeetMate - AI Meeting Notes", layout="centered")
st.title("📂 MeetMate - AI-Powered Meeting Summarizer")
st.markdown("Convert your meeting recordings into clean, structured notes in just a few clicks.")

st.divider()

# File uploader
st.subheader("🔊 Upload Your Audio File")
uploaded_file = st.file_uploader(
    "Supported formats: MP3, WAV, MP4", type=["mp3", "wav", "mp4"]
)

# Transcription and summarization logic
if uploaded_file is not None:
    with st.spinner("🔄 Processing audio and generating notes..."):
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        # Transcribe audio
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1)

        with open(temp_audio_path, "rb") as audio_file:
            transcription = transcriber.transcribe(audio_file, config)

        if transcription.status == aai.TranscriptStatus.error:
            st.error(f"❌ Transcription failed: {transcription.error}")
        else:
            st.success("✅ Transcription completed successfully!")
            st.subheader("📝 Transcribed Text")
            st.text_area("Full Transcript", transcription.text, height=300)

            # generate meeting notes with Gemini
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are an AI assistant that receives transcribed meeting text and generates well-organized "
                "meeting notes with bullet points, timelines, key takeaways, and action items. "
                f"Here is the transcript:\n\n{transcription.text}"
            )
            response = model.generate_content(prompt)

            # display notes
            st.subheader("📄 Generated Meeting Notes")
            st.markdown(response.text)

else:
    st.info("📥 Please upload an audio file to get started.")
