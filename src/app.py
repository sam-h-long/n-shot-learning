import streamlit as st
import tiktoken
import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv
from enum import Enum

# Title of the app
st.set_page_config(layout='wide')
st.title("Zero Shot Learning Demo")
st.caption("using OpenAI's GPT-4 model")

# Load the environment variables
load_dotenv(override=True)


@st.cache_resource
def get_azure_openai_client():
    return openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-10-21",  # https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


class ClassificationLabelsKatja(Enum):
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    LUNG = "lung"
    KIDNEY = "kidney"
    LIVER = "liver"
    DIGESTIVE = "digestive"
    CANCER = "cancer"
    DERMATOLOGICAL = "dermatological"
    RHEUMATOLOGICAL = "rheumatological"
    EYE = "eye"
    EARNOSETHROAT = "ear nose throat"
    MENTAL = "mental"
    PHYSIOTHERAPY = "physiotherapy"
    DOCTOR_TREATMENT = "doctor treatment"
    HOSPITAL_TREATMENT = "hospital treatment"
    FATIGUE = "fatigue"
    MUSCULOSCELETAL = "musculoskeletal"
    MEDICATION = "medication"


def _identify_label_collision(token_list: list, label_value: str, collision_dict: dict, token_index: int = 0):
    # Check for collisions in the nth token
    if token_list[token_index] in collision_dict:
        print(f"COLLISION TOKEN {token_index}: {token_list[token_index]}")
        print(f"\t token {token_index}: {token_list[token_index]}")
        print(f"\t collision label: {label_value}")
        print(f"\t existing  label: {collision_dict[token_list[token_index]]}")
    else:
        collision_dict[token_list[token_index]] = label_value
    return collision_dict


def _highlight_duplicates_ignore_blanks_written_by_chatgpt(df):
    def style_column(col):
        duplicates = col.duplicated(keep=False) & (col != "")
        return ["background-color: yellow" if duplicates.iloc[i] else "" for i in range(len(col))]

    return df.style.apply(style_column, axis=0)


def display_tokenization(tokenizer, label_class=ClassificationLabelsKatja):
    # Get the tokenization, subwords, and length of the tokenization
    label_sub_words_dict = dict()
    for label in label_class:
        tokens = tokenizer.encode(label.value)
        sub_words = list()
        for tok in tokens:
            sub_words.append(tokenizer.decode([tok]))
        print(label.value, " -> ", tokens, " = ", sub_words, " <- ", len(tokens), " tokens")
        label_sub_words_dict[label.value] = sub_words
    max_tokens_in_label = max([len(sub_words) for sub_words in label_sub_words_dict.values()])

    # Display the tokenization, subwords, and length of the tokenization
    show_df = pd.DataFrame.from_dict(label_sub_words_dict, orient='index')
    show_df = show_df.fillna("")
    styled_df = _highlight_duplicates_ignore_blanks_written_by_chatgpt(df=show_df)
    st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

    ### MIGHT DELETE THIS PART  -------------------------
    # Check for collisions dictionary of dictionaries
    master_collision_dict = dict()
    for mn in range(max_tokens_in_label):
        master_collision_dict[mn] = dict()

    for label, tokens in label_sub_words_dict.items():
        for i in range(len(tokens)):
            master_collision_dict[i] = _identify_label_collision(token_list=tokens, label_value=label,
                                                                 collision_dict=master_collision_dict[i], token_index=i)
    master_collision_reverse_dict = dict()
    for key, original_dict in master_collision_dict.items():
        master_collision_reverse_dict[key] = {v: k for k, v in original_dict.items()}
    ### --------------------------------------------------

    return label_sub_words_dict, max_tokens_in_label, master_collision_reverse_dict


def display_classification(choice):
    category = choice.message.content
    st.write(f"**Suggested Classification:** {category}")

    category_tokens = tokenizer_tiktoken.encode(category)

    sub_words = list()
    for tok in category_tokens:
        sub_words.append(tokenizer_tiktoken.decode([tok]))

    columns = st.columns(len(category_tokens))
    for i, col in enumerate(columns):
        with col:
            token_i = choice.logprobs.content[i]
            token_i_prob = np.round(np.exp(token_i.logprob) * 100, 2)

            st.write(f"**Token {i}:** ", token_i.token)
            st.write(f"**Prob( {token_i.token} | {sub_words[0:i]} )** = {token_i_prob}%")

            token_i_dict = {"Token": list(),
                            "Log Probability": list(),
                            "Probability": list(),
                            "Is in Labels?": list()}
            for j, top_log_j in enumerate(token_i.top_logprobs):
                token_i_dict["Token"].append(top_log_j.token)
                token_i_dict["Log Probability"].append(top_log_j.logprob)
                token_i_dict["Probability"].append(np.round(np.exp(top_log_j.logprob) * 100, 2))
                token_i_dict["Is in Labels?"].append(top_log_j.token in master_c_reverse_dict[i].values())
                # token_i_dict["Selected?"].append(top_log_j.token == token_i.token)

            df = pd.DataFrame(token_i_dict)
            df["Probability"] = df["Probability"].map(lambda x: f"{x:.2f}")
            df = df.drop(columns=["Log Probability"])

            def highlight_selected(row, actual=token_i.token):
                if row["Token"] == actual:  # Check to avoid errors if 'Selected?' column is dropped
                    return ["background-color: yellow"] * len(row)
                else:
                    return [""] * len(row)

            styled_df = df.style.apply(highlight_selected, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.json(token_i_dict, expanded=False)


st.divider()
st.header("Classification Labels Tokenized:")
MODEL_NAME = "gpt-4o"
tokenizer_tiktoken = tiktoken.encoding_for_model(model_name=MODEL_NAME)
st.caption(f"using {tokenizer_tiktoken} as the tokenizer")
label_sw_dict, max_tks_in_label, master_c_reverse_dict = display_tokenization(tokenizer_tiktoken, label_class=ClassificationLabelsKatja)

st.divider()
st.header("Classify a Term:")

SYSTEM_PROMPT = "You are a classifier that categorizes medical terms based purely on their most likely match within the provided categories. Here is a list of categories and I will provide a term to classify:\n\nCategories:\n"
for i, category in enumerate(label_sw_dict.keys(), start=1):
    SYSTEM_PROMPT += f"{i}. {category}\n"
SYSTEM_PROMPT += f"\nOutput only the category name that the term belongs to."

system_prompt = st.text_area("**System Prompt:**", value=SYSTEM_PROMPT, height=550, placeholder="Type your system prompt here...")

USER_PROMPT = "term: "
user_prompt = st.text_area("**User Prompt:**", value=USER_PROMPT, height=68, placeholder="Type your user prompt here...")

if st.button("Run Classification"):
    client = get_azure_openai_client()

    if USER_PROMPT == user_prompt:
        st.error("Please provide a term. *For example, 'term: allergen av gjærsopp, cladosporium herbarum'*")
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            model="d-storebot-openai-gpt4o",
            messages=messages,
            max_tokens=max_tks_in_label,
            seed=42,
            temperature=0.2,
            n=2,
            logprobs=True,
            top_logprobs=5
        )
        st.divider()
        for c_i in range(len(response.choices)):
            st.subheader(f"**Classification {c_i}:**")
            display_classification(response.choices[c_i])
            st.divider()
