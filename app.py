import streamlit as st
import assemblyai as aai
from dotenv import load_dotenv
import os
import tempfile
import google.generativeai as genai
print(genai.__file__)




# api settings 
load_dotenv(override=True)
aai.settings.api_key=os.getenv('ASS_API_KEY')
genai.configure(api_key=os.getenv("GEN_AI_KEY"))


st.title("upload audio file ")
st.subheader("Input audio")

# audio upload
uploaded_file = st.file_uploader("upload meeting audio",type=["mp3","wav","mp4"])
print (uploaded_file)

# set assemblyai transcriber 
transcriber=aai.Transcriber()
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1)

if uploaded_file is not  None:


    with st.spinner("Processing audio..."):
        # read audio file 
        with tempfile.NamedTemporaryFile(delete=False,suffix=".mp3") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path=temp_audio.name

        
        # transcribe 
        with open(temp_audio_path, "rb") as audio_file:
            transcription =transcriber.transcribe(audio_file,config)
        if transcription.status == aai.TranscriptStatus.error:
            st.error(f"Transcription failed: {transcription.error}")

        st.success("Transcription complete!")
        st.text_area("Transcribed Text", transcription.text, height=300)
        
        model=genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
                f"You are an AI assistant that receives transcribed meeting text and generates fully "
                f"organized meeting notes with bullet points, timelines, and all important information. "
                f"Here is the transcript:\n\n{transcription.text}"
        )

        response=model.generate_content(prompt)

        st.subheader("meeting notes")
        st.markdown(response.text)

        # 


else:
    st.info('upload an audio file ')





