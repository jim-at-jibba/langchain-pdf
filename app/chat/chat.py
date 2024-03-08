import random
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.llms.chatopenai import build_llm
from app.chat.memories.sql_memory import build_memory
from app.web.api import set_conversation_components, get_conversation_components


def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """
    components = get_conversation_components(chat_args.conversation_id)
    previous_retriever = components["retriever"]
    retriever = None
    if previous_retriever:
        # this is NOT the first message of the conversation and
        # I need to use the same retriever
        build_retriever = retriever_map[previous_retriever]
        retriever = build_retriever(chat_args)
    else:
        # this is the first and I need to pick a random retriever
        random_retriever_name = random.choice(list(retriever_map))
        build_retriever = retriever_map[random_retriever_name]
        retriever = build_retriever(chat_args)
        set_conversation_components(
            conversation_id=chat_args.conversation_id,
            llm="",
            memory="",
            retriever=random_retriever_name,
        )

    llm = build_llm(chat_args)
    # A second llm is created for use with the condense uestion chain to help
    # with the fact that callbacks are shared across all objects in a chain
    # By using a separate llm we can set streaming to false and then filter based on
    # that property in the callback
    condense_question_llm = ChatOpenAI(streaming=False)
    memory = build_memory(chat_args)

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        condense_question_llm=condense_question_llm,
    )
