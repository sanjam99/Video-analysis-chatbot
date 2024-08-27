import streamlit as st
import os
import os.path

from dotenv import load_dotenv
from llama_index.response.pprint_utils import pprint_response
from llama_index.llms import OpenAI
from llama_index import download_loader
from llama_index import VectorStoreIndex, load_index_from_storage, ServiceContext
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_hub.youtube_transcript import is_youtube_video
from moviepy.editor import VideoFileClip
import whisper

from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.ollama import OllamaMultiModal

from pydub import AudioSegment
from pydub.utils import make_chunks
# for sh user For API usage in React and Nodejs app
class VideoProcessor:
    def __init__(self, video_path, output_audio_path, image_path, text_path):
        self.video_path = video_path
        self.output_audio_path = output_audio_path
        self.image_path = image_path
        self.text_path = text_path

    def extract_audio(self):
        video = VideoFileClip(os.path.join(self.video_path, "video.mp4"))
        audio_part = video.audio
        audio_part.write_audiofile(os.path.join(self.output_audio_path, "output_audio.mp3"))

    def segment_audio(self):
        audio = AudioSegment.from_mp3(os.path.join(self.output_audio_path, "output_audio.mp3"))
        chunk_length_ms = 2000
        chunks = make_chunks(audio, chunk_length_ms)
        for i, chunk in enumerate(chunks):
            chunk_name = os.path.join(self.output_audio_path, f"{i}.mp3")
            chunk.export(chunk_name, format="mp3")
        os.remove(os.path.join(self.output_audio_path, "output_audio.mp3"))

    def extract_text(self):
        model = whisper.load_model("base.en")
        audio_text = ''
        for filename in os.listdir(self.output_audio_path):
                file_path = os.path.join(self.output_audio_path, filename)
                result = model.transcribe(file_path)
                time = int(filename[:-4]) * 2
                audio_text += str(f'At time {time}s:') + result['text'] + '\n'
        with open(os.path.join(self.text_path, "audio.md"), "w") as file:
            file.write(audio_text)
            file.close()

    def extract_frames(self):
            clip = VideoFileClip(os.path.join(self.video_path, "video.mp4"))
            clip.write_images_sequence(os.path.join(self.image_path, "%04d.png"), fps=0.5)

    def process_video(self):
        self.extract_audio()
        self.segment_audio()
        self.extract_text()
        self.extract_frames()


class translate_image_to_text:
    def __init__(self, image_path, text_path):
        self.image_path = image_path
        self.text_path = text_path
        self.response = ''
    def get_image_path(self):
        image_folder = self.image_path
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        return image_files
    def image_to_text(self):
        mm_model = OllamaMultiModal(model='llava', temperature=0) 
        image_file_names = self.get_image_path()
        for image in image_file_names:
            print(image)
            time = 2*int(image[8:-4])
            self.response += str(f'At time {time}s:')+ str(mm_model.complete(prompt='summarize the image and output as markdown format with one line', image_documents=[ImageDocument(image_path=image)])) + '\n' 
        with open(self.text_path+'image.md', 'w') as file:
                file.write(self.response)
                file.close()

    def reply(self):
        # embed_model = TextEmbeddingsInference(model_name="BAAI/bge-large-en-v1.5")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        llm = Ollama(model='llava', request_timeout=100, temperature=0)
        Settings.llm = llm
        Settings.embed_model = embed_model
        data = SimpleDirectoryReader(text_path).load_data()
        index = VectorStoreIndex.from_documents(data)
        query_engine = index.as_query_engine(similarity_top_k=3)
        while True:
            try:
                user_input = input('\033[94m' +"Prompt: " + '\033[0m')
                response = query_engine.query(user_input)
                print(response)
            except KeyboardInterrupt:
                break

if __name__ == '__main__':

    video_path = './video/'
    output_audio_path = './audio/'
    image_path = './image/'
    text_path = './text/'
    # output_folder= './output/'
# process video to images and text
    processor = VideoProcessor(video_path, output_audio_path, image_path, text_path)
    processor.process_video()

    text = translate_image_to_text(image_path=image_path, text_path=text_path)
    text.image_to_text()
    text.reply()


# beyaond this is streamlit app process for initial review 
load_dotenv()

storage_path = "./vectorstore"

llm = OpenAI(temperature=0.1, model="gpt-4-turbo-preview")
service_context = ServiceContext.from_defaults(llm=llm)

documents = None

with st.sidebar:
    st.title("Youtube")
    urlTextValue = st.text_input(label="Youtube URL")
    st.button(label="Load URL", on_click=lambda: loadYoutubeURL(urlTextValue))

st.title("Ask the Youtube")
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question !"}
    ]


def cleanIndex():
    global documents
    if documents != None :
        st.write("Found Index and Vectors, Cleaning up now...")
        ids = [str(i) for i in range(1, len(docs) + 1)]
        index = VectorStoreIndex.from_documents(documents)
        docsToDelete = index.similarity_search("")
        print(docsToDelete[0].metadata)
        st.write("Count before cleanup", documents._collection.count())
        docsToDelete._collection.delete(ids=[ids[-1]])
        st.write("count after cleanup", documents._collection.count())

def loadYoutubeURL(url):
    global documents
    cleanIndex()
    if is_youtube_video(url) == True :
        with st.spinner("Loading the Index..."):
            print(url)
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(ytlinks=[url])
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=storage_path)
            chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True, verbose=True)
            print(chat_engine)
            st.session_state["chat_engine"] = chat_engine
    else :
        st.error("Please check the youtube URL, it doesn't seem to be valid", icon="🚨")


if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("Prompt recieved")
            chat_engine = st.session_state["chat_engine"]
            print(chat_engine)
            if chat_engine != None :
                response = chat_engine.chat(prompt)
                st.write(response.response)
                pprint_response(response, show_source=True)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) 
            else :
                st.write("Please load a youtube video first...")