import os
import streamlit as st
from openai import OpenAI
import re
from typing import List, Tuple
import dotenv
from difflib import SequenceMatcher

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GUIDELINES = """
Guidelines() {
return (
<>
- Only apply the change(s) suggested by the most recent assistant message
(before your generation)
<br />
- Do not make any unrelated changes to the code
<br />
- Produce a valid full rewrite of the entire original file without
skipping any lines. Do not be lazy!
<br />
- Do not arbitrarily delete pre-existing comments/empty lines
<br />
- Do not omit large parts of the original file for no reason
<br />
- Do not omit any needed changes from the requisite messages/code blocks
<br />
- If there is a clicked code block, bias towards just applying that
(and applying other changes implied)
<br />
- In the final output, provide only the code without any backticks, language specifiers, or other formatting
</>
);
}
"""

def get_streaming_completion(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 2000, 
                             temperature: float = 0.7) -> str:
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GUIDELINES},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        result = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content
        return result.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

def tokenize(text: str) -> List[str]:
    return re.findall(r'\w+|[^\w\s]', text)

def detokenize(tokens: List[str]) -> str:
    return ' '.join(tokens).replace(' .', '.').replace(' ,', ',')

def speculative_edit(code: str, instructions: str, num_tokens: int = 10, temperature: float = 0.7) -> str:
    tokens = tokenize(code)
    final_tokens = tokens.copy()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(tokens), num_tokens):
        context = detokenize(final_tokens)
        prompt = f"Continue this code, considering the following instructions: {instructions}\n\n{context}"
        speculation = get_streaming_completion(prompt, max_tokens=num_tokens, temperature=temperature)
        
        spec_tokens = tokenize(speculation)
        actual_next = tokens[i:i+len(spec_tokens)]
        
        correct_tokens, incorrect_index = compare_tokens(spec_tokens, actual_next)
        final_tokens.extend(correct_tokens)
        
        progress = min(1.0, i / len(tokens))
        progress_bar.progress(progress)
        status_text.text(f"Processing... {int(progress * 100)}%")
        
        if incorrect_index < len(spec_tokens):
            break
    
    final_code = detokenize(final_tokens)
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    return validate_and_improve(final_code, instructions)

def compare_tokens(spec_tokens: List[str], actual_tokens: List[str]) -> Tuple[List[str], int]:
    correct_tokens = []
    for i, (spec, actual) in enumerate(zip(spec_tokens, actual_tokens)):
        if spec.lower() == actual.lower():
            correct_tokens.append(spec)
        else:
            return correct_tokens, i
    return correct_tokens, len(correct_tokens)

def validate_and_improve(code: str, instructions: str) -> str:
    validation_prompt = f"Improve the following code if needed, considering these instructions: {instructions}. If it's already correct and complete, return it as is. Do not include any backticks, language specifiers, or other formatting:\n\n{code}"
    improved_code = get_streaming_completion(validation_prompt, max_tokens=2000, temperature=0.5)
    return remove_formatting(improved_code)

def remove_formatting(code: str) -> str:
    # Remove backticks and language specifiers
    code = re.sub(r'^```[\w]*\n|```$', '', code, flags=re.MULTILINE)
    return code.strip()

def highlight_diff(old_code: str, new_code: str) -> str:
    differ = SequenceMatcher(None, old_code, new_code)
    highlighted_code = ""
    for opcode, i1, i2, j1, j2 in differ.get_opcodes():
        if opcode == 'equal':
            highlighted_code += old_code[i1:i2]
        elif opcode == 'delete':
            highlighted_code += f'<span style="background-color: #ffcccc;">{old_code[i1:i2]}</span>'
        elif opcode == 'insert':
            highlighted_code += f'<span style="background-color: #ccffcc;">{new_code[j1:j2]}</span>'
        elif opcode == 'replace':
            highlighted_code += f'<span style="background-color: #ffcccc;">{old_code[i1:i2]}</span>'
            highlighted_code += f'<span style="background-color: #ccffcc;">{new_code[j1:j2]}</span>'
    return highlighted_code.replace('\n', '<br>')

def detect_language(code: str) -> str:
    # This is a simple language detection. You might want to use a more sophisticated method.
    if 'def ' in code or 'import ' in code:
        return 'python'
    elif 'function ' in code or 'var ' in code or 'let ' in code or 'const ' in code:
        return 'javascript'
    elif 'public class ' in code or 'public static void main' in code:
        return 'java'
    else:
        return 'text'

def add_comments(code: str) -> str:
    prompt = f"Add detailed and helpful comments to the following code. Ensure the comments are informative and explain the purpose of each section or important line:\n\n{code}"
    commented_code = get_streaming_completion(prompt, max_tokens=2000, temperature=0.5)
    return remove_formatting(commented_code)

def explain_code(code: str) -> str:
    prompt = f"Provide a detailed explanation of the following code, including its purpose, structure, and key components:\n\n{code}"
    explanation = get_streaming_completion(prompt, max_tokens=2000, temperature=0.5)
    return explanation

def summarize_code(code: str) -> str:
    prompt = f"Provide a concise summary of the following code, highlighting its main functionality and purpose:\n\n{code}"
    summary = get_streaming_completion(prompt, max_tokens=1000, temperature=0.5)
    return summary

def main():
    st.set_page_config(page_title="AI-Powered Code Editor", layout="wide")
    
    # Custom CSS to improve UI
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        color: #333333;
        font-size: 16px;
    }
    .stMarkdown {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 AI-Powered Code Editor")
    st.write("Enter your code below, provide instructions, and use the buttons to improve, comment, explain, or summarize your code.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        initial_code = st.text_area("Enter your code here:", height=400)
    
    with col2:
        instructions = st.text_area("Enter instructions for code improvement:", height=200)
    
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        if st.button("Improve Code"):
            if initial_code:
                with st.spinner("Improving code..."):
                    improved_code = speculative_edit(initial_code, instructions)
                    highlighted_diff = highlight_diff(initial_code, improved_code)
                st.markdown("### Differences:")
                st.markdown(highlighted_diff, unsafe_allow_html=True)
                st.markdown("### Improved Code:")
                language = detect_language(improved_code)
                st.code(improved_code, language=language, line_numbers=True)
            else:
                st.warning("Please enter some code to improve.")
    
    with col4:
        if st.button("Add Comments"):
            if initial_code:
                with st.spinner("Adding comments..."):
                    commented_code = add_comments(initial_code)
                st.markdown("### Commented Code:")
                st.code(commented_code, language=detect_language(commented_code), line_numbers=True)
            else:
                st.warning("Please enter some code to comment.")
    
    with col5:
        if st.button("Explain Code"):
            if initial_code:
                with st.spinner("Generating explanation..."):
                    explanation = explain_code(initial_code)
                st.markdown("### Code Explanation")
                st.write(explanation)
            else:
                st.warning("Please enter some code to explain.")
    
    with col6:
        if st.button("Summarize Code"):
            if initial_code:
                with st.spinner("Generating summary..."):
                    summary = summarize_code(initial_code)
                st.markdown("### Code Summary")
                st.write(summary)
            else:
                st.warning("Please enter some code to summarize.")

if __name__ == "__main__":
    main()