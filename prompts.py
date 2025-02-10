from config import SystemPromptEnum

def get_sys_prompt(sys: SystemPromptEnum) -> str:
    if sys == SystemPromptEnum.none:
        return ""
    elif sys == SystemPromptEnum.task:
        return "\nAnswer the question by thinking step by step. At the end, output the final answer, starting with 'Answer:'."
    else:
        raise ValueError(f"unsupported system prompt: {sys}")

def get_self_rewarding_text_prompt(question: str, response: str) -> str:
    text = f"""Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to 
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question, 
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a 
useful way, regardless of whether it seems to have been written by an AI Assistant or if it 
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {question}

<response>{response}</response>

After examining the user’s instruction and the response:

- Briefly justify your total score, up to 100 words. 
- Conclude with the score using the format: “Score: <total points>”

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria.
"""
    return text

if __name__ == "__main__":
    print(get_self_rewarding_text_prompt())
