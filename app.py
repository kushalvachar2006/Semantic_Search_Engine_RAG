import streamlit as st
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


dotenv.load_dotenv()

st.title("Semantic Search Engine")
st.header("Upload a file to get started",divider="green")

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

#Vector store
chroma_vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embedding_model,
    persist_directory='./chroma/db'
)

# LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0) 

uploaded_file=st.file_uploader("Select a file:")




if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            print("File info: ",uploaded_file)

        # save file in memory
            temp_file_path = uploaded_file.name
            with open(temp_file_path,"wb") as f:
                f.write(uploaded_file.getbuffer())

        # PDF file loader
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            # print("Docs: ",docs)

        # create chunk
            chunks = text_splitter.split_documents(docs)
            # print("Chunks created: ",chunks)

            # for i, chunk in enumerate(chunks):
            #     print(f"Chunk {i} is of size {len(chunk.page_content)} characters.")

        # create embeddings
            # emb1 = embedding_model.embed_query(chunks[0].page_content)
            # print(emb1)

        # Index embedding
            
            chroma_idx = chroma_vector_store.add_documents(documents=chunks)
            #print("Documents indexed in Chroma vector store.",chroma_idx)

        # Similarity search
            # result = chroma_vector_store.similarity_search("What are the main topics of the document?")
            # print("Similarity search result: ",result)
            
            retriever = chroma_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})   
            if prompt := st.chat_input("Prompt"):
                print(prompt)

                docs_retrieved = retriever.invoke(prompt)

        # Create prompt template 
                systemPrompt = """
                You are a helpful assistant. Please answer the following question
                {question}, only using the following information {document}.
                If you can't answer the question, please say "I don't know".
                """

                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", systemPrompt),
                        ("human", "{question}"),
                    ]
                )

                context = "\n\n".join(
                           doc.page_content for doc in docs_retrieved
                        )

                final_prompt = prompt_template.invoke(
                    {
                        "question": prompt,
                        "document": context
                    }
                )
                print(f"Final prompt: {final_prompt}")

                # UI container
                result_placeholder = st.empty()

                completion = llm.invoke(final_prompt)
                print(completion.content)
                print(f"Completion: {completion.content}")
                #st.chat_message("assistant").write(completion)

                # Streaming the completion result
                full_completion = ""
                for chunk in llm.stream(final_prompt):
                    full_completion += chunk.content
                    result_placeholder.write(full_completion)

            



        except Exception as e:
            print(e)