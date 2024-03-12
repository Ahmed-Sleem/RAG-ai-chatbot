from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


class Chatbot:
    def __init__(self, data, key, type=0):
        self.type = type
        self.extracted_text = ""
        self.pdf_docs = data
        self.vectorstore = None
        self.conversation_chain = None
        self.HUGGINGFACEHUB_API_TOKEN = key

    def get_pdf_text(self):
        text = ""
        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        self.extracted_text = text
        return text

    def get_text_chunks(self):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(self.extracted_text)
        if len(chunks) > 1000:
            print('Need to tune the separation of the chunks')
        return chunks

    def get_vectorstore(self, chunks):
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 1}
        )
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore

    def get_conversation_chain(self, vectorstore):
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.8, "max_length": 512},
            huggingfacehub_api_token=self.HUGGINGFACEHUB_API_TOKEN
        )
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    def handle_user_input(self, user_question, conversation):
        response = conversation({'question': user_question})
        chat_history = response['chat_history']
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                print(f"User: {message.content}")
                return f"User: {message.content}"
            else:
                print(f"Bot: {message.content}")
                return f"User: {message.content}"

    def start(self):
        if self.type == 0:
            pdf_docs = self.pdf_docs
            raw_text = self.get_pdf_text()
            text_chunks = self.get_text_chunks()
            vectorstore = self.get_vectorstore(text_chunks)
            conversation = self.get_conversation_chain(vectorstore)
            self.conversation_chain = conversation
        elif self.type == 1:
            raw_text = self.extracted_text
            text_chunks = self.get_text_chunks()
            vectorstore = self.get_vectorstore(text_chunks)
            conversation = self.get_conversation_chain(vectorstore)
            self.conversation_chain = conversation


def main():
    
    
    
    
    # Your data
    key = ''  # HUGGINGFACEHUB_API_TOKEN
    pdf_docs = [""]  # PDF or text data




    # Setting up the bot
    bot = Chatbot(pdf_docs, key, 0)  # Set up your data and API key
    bot.start()  # Start adding the data to the bot and prepare it for running




    # Run question and answer
    while True:
        user_question = input("Ask a question (type 'exit' to quit): ")  # User input

        if user_question.lower() == 'exit':
            break
        response = bot.handle_user_input(user_question, bot.conversation_chain)


if __name__ == '__main__':
    main()
