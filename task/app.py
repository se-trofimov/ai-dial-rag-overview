import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
try:
    from task._constants import DIAL_URL, API_KEY
except ModuleNotFoundError:
    from _constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")
        folder_path = os.path.join(os.path.dirname(__file__), "microwave_faiss_index")

        if os.path.exists(folder_path):
            print("✅ Found existing FAISS index. Loading from local storage...")
            return FAISS.load_local(
                folder_path=folder_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

        print("🆕 No local FAISS index found. Creating a new one...")
        return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        manual_path = os.path.join(os.path.dirname(__file__), "microwave_manual.txt")
        loader = TextLoader(file_path=manual_path, encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
        )
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(os.path.join(os.path.dirname(__file__), "microwave_faiss_index"))
        print("💾 New FAISS index has been created and saved locally.")
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score,
        )

        context_parts = []
        for index, (doc, relevance_score) in enumerate(results, start=1):
            context_parts.append(doc.page_content)
            print(f"\nResult #{index} | Score: {relevance_score:.4f}")
            print(doc.page_content)

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        print(response.content)
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        context = rag.retrieve_context(user_question)
        augmented_prompt = rag.augment_prompt(user_question, context)
        rag.generate_answer(augmented_prompt)



main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version="",
        ),
    )
)