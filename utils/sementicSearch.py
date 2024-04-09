import os
from langchain.embeddings.openai import OpenAIEmbeddings

# from langchain.llms import openai
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# import openai

# from langchain.llms import openai
from pymilvus import connections, Collection
from dotenv import load_dotenv

load_dotenv("../.env")

db = connections.connect(
    user="root", password="jarvis@admin", host="192.168.1.98", port="19530"
)
collection = Collection(name="gptTeacher", using="default")

embeddingd_api = os.getenv("OPENAI_EMBEDINGS_API_KEY")
embeddings = OpenAIEmbeddings(
    openai_api_key=embeddingd_api, model="text-embedding-ada-002"
)

# llm = openai.OpenAI(
#     model="gpt-3.5-turbo", openai_api_key=embeddingd_api, max_tokens=1024
# )

llm = ChatOpenAI(
    model_name='gpt-4',
    openai_api_key = embeddingd_api,      
    max_tokens=2048,
)



def find_sementic_docs(text: str):
    """Searches the vector database for the given query and finds out relevant documents"""

    embedding = embeddings.embed_query(text)

    results = collection.search(
        data=[embedding],
        anns_field="ada_embedings",
        param={"metric_type": "L2", "top_k": 10},
        limit=10,
        output_fields=["page_num", "text"],
    )

    return [result.entity.get("text") for result in results[0]]


def ask_llm(text: str):
    """Use GPT-3 to ask questions from relevant documents"""

    relevant_docs = find_sementic_docs(text)

    system_template = """You are a Teacher which answers questions os students. You are given a three most relevent passage of text and a question. You have to answer the question based on the passage of text. You can use your knowledge. The passage of text is given below:\n\n"""
    # system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = f"""
    Passage 1:\n\n {relevant_docs[0]}\n\n Passage 2:\n\n {relevant_docs[1]}\n\n Passage 3:\n\n {relevant_docs[2]}\n\n
    Question: {text}\n Answer:
    """
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # chat_prompt = ChatPromptTemplate.from_messages(
    #     [system_message_prompt, human_message_prompt]
    # )

    prompt = system_template+human_template
    print(prompt)

    # openai.api_key = embeddingd_api
    # reply = openai.ChatCompletion.create(
    #     engine="text-davinci-003",
    #     messages=[
    #         {"role": "system", "content": system_template},
    #         {"role": "user", "content": human_template},
    #     ],
    #     max_tokens=1024,
    #     # api_key=embeddingd_api,
    # )
    
    reply = llm(
        messages=[
            SystemMessage(content=system_template),
            HumanMessage(content=human_template),
        ]
    )
    
    # reply = llm(system_template+human_template)

    return reply.content
