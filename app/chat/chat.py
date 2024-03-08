import random
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.llms import llm_map
from app.chat.memories import memory_map
from app.web.api import set_conversation_components, get_conversation_components


def select_component(component_type, component_map, chat_args):
    components = get_conversation_components(chat_args.conversation_id)

    previous_component = components[component_type]

    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)

    else:
        random_name = random.choice(list(component_map.keys()))
        builder = component_map[random_name]
        return random_name, builder(chat_args)


def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """
    retriver_name, retriever = select_component(
        "retriever",
        retriever_map,
        chat_args,
    )

    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args,
    )

    memory_name, memory = select_component(
        "memory",
        memory_map,
        chat_args,
    )

    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriver_name,
        memory=memory_name,
    )

    print(
        f"Running chain with: \n\n memory: {memory_name}, llm: {llm_name}, retriver: {retriver_name}"
    )

    # A second llm is created for use with the condense uestion chain to help
    # with the fact that callbacks are shared across all objects in a chain
    # By using a separate llm we can set streaming to false and then filter based on
    # that property in the callback
    condense_question_llm = ChatOpenAI(streaming=False)

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        condense_question_llm=condense_question_llm,
    )
