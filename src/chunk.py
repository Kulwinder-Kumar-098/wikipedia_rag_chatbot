import json
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dir='C:\\Users\\hp\\OneDrive\\Desktop\\The_Wikipedia_RAG_ChatBot\\data\\processed'
CHUNK_SIZE = 500     
CHUNK_OVERLAP = 50    

def chunk_with_langchain(file_path, chunk_size= CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    loader = DirectoryLoader(dir,
                             glob="*.txt",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"} 
                             )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Split the loaded documents into chunks
    split_docs = text_splitter.split_documents(documents)
    
    # Extract just the string content from LangChain's Document objects
    return [doc.page_content for doc in split_docs]


if __name__ == "__main__":
    chunks = chunk_with_langchain(dir, CHUNK_SIZE, CHUNK_OVERLAP)
    
    print(f"Total chunks created: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
    
    if len(chunks) > 0:
        print("---- First chunk ----")
        print(chunks[0])
    if len(chunks) > 1:
        print("\n---- Second chunk (note the overlap at the start) ----")
        print(chunks[1])

    # Save chunks as JSON: list of {"id": int, "text": str}
    records = [{"id": i, "text": c} for i, c in enumerate(chunks)]
    out_path = "C:\\Users\\hp\\OneDrive\\Desktop\\The_Wikipedia_RAG_ChatBot\\data\\processed\\chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} chunks to {out_path}")
