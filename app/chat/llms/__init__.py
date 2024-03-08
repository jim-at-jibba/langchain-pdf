from functools import partial
from .chatopenai import build_llm

# partial allows us to use partial application
# it creates a new function with accepts parameters plus the
# ones passed in when we created the partial
# means we can reuse the build_llm function and pass in params ahead of time
llm_map = {
    "gpt-4": partial(build_llm, model_name="gpt-4"),
    "gpt-3.5-turbo": partial(build_llm, model_name="gpt-3.5-turbo"),
}
