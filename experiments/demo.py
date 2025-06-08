import pandas as pd
import streamlit as st
import spacy
import unicodedata

from cassis import Cas, load_typesystem
from rwse_checker.rwse import RWSE_Checker, T_RWSE, T_SENTENCE, T_TOKEN
from transformers.utils import logging

logging.set_verbosity_error()

STATE_CAS = 'STATE_CAS'
STATE_CHECKBOX_ = 'STATE_CHECKBOX_'
STATE_CONFUSION_SETS_MODIFIED = 'STATE_CONFUSION_SETS_MODIFIED'
STATE_CONFUSION_SETS_ORIGINAL = 'STATE_CONFUSION_SETS_ORIGINAL'
STATE_INFINITE_SEQUENCE = 'STATE_INFINITE_SEQUENCE'
STATE_RWSE_CHECKER = 'STATE_RWSE_CHECKER'
STATE_TEXT_AREA = 'STATE_TEXT_AREA'

nlp = spacy.load("en_core_web_sm")

ts_file = 'experiments/input/TypeSystem.xml'
with open(ts_file, 'rb') as f:
    ts = load_typesystem(f)

if st.session_state.get(STATE_INFINITE_SEQUENCE) is None:
    def infinite_sequence():
        num = 0
        while True:
            yield num
            num += 1

    st.session_state[STATE_INFINITE_SEQUENCE] = infinite_sequence()

if st.session_state.get(STATE_CONFUSION_SETS_ORIGINAL) is None:
    st.session_state[STATE_CONFUSION_SETS_ORIGINAL] = []
    with open('experiments/input/confusion_sets_modified.csv', 'r') as file:
        for line in file.readlines():
            st.session_state[STATE_CONFUSION_SETS_ORIGINAL].append({item.strip() for item in line.split(',')})

if st.session_state.get(STATE_CONFUSION_SETS_MODIFIED) is None:
    st.session_state[STATE_CONFUSION_SETS_MODIFIED] = []
    for item in st.session_state[STATE_CONFUSION_SETS_ORIGINAL]:
        cs_id = next(st.session_state[STATE_INFINITE_SEQUENCE])
        st.session_state[STATE_CONFUSION_SETS_MODIFIED].append((item, cs_id))
        st.session_state[STATE_CHECKBOX_ + str(cs_id)] = True

if st.session_state.get(STATE_RWSE_CHECKER) is None:
    st.session_state[STATE_RWSE_CHECKER] = RWSE_Checker()
    st.session_state[STATE_RWSE_CHECKER].set_confusion_sets(st.session_state[STATE_CONFUSION_SETS_ORIGINAL])

def reset_confusion_sets():
    st.session_state[STATE_RWSE_CHECKER].set_confusion_sets([
        item[0] for item in st.session_state[STATE_CONFUSION_SETS_MODIFIED] if
        st.session_state[STATE_CHECKBOX_ + str(item[1])]
    ])

default_text = "My advise for you is: Do not put to much subjects, just put a few subject and make them look interesting."
# I sailed over the pony.
# It's ease too dessert the county.

labels_to_colors = {
    'RWSE': 'palegreen',
    'CORR': 'limegreen',
}

def clean_string(text: str) -> str:
    cleaned = ''.join(ch if unicodedata.category(ch)[0] != "C" else ' ' for ch in text.strip())
    cleaned = cleaned.replace('´', '\'')
    cleaned = cleaned.replace('`', '\'')
    return ' '.join(cleaned.split())

def create_cas(input_text):
    cas = Cas(ts)
    cas.sofa_string = clean_string(input_text)

    doc = nlp(cas.sofa_string)
    S = ts.get_type(T_SENTENCE)
    for sent in doc.sents:
        cas_sentence = S(begin=sent.start_char, end=sent.end_char)
        cas.add(cas_sentence)

    T = ts.get_type(T_TOKEN)
    for token in doc:
        cas_token = T(begin=token.idx, end=token.idx + len(token.text))
        cas.add(cas_token)

    # run RWSE checker
    st.session_state[STATE_RWSE_CHECKER].check_cas(cas, ts)
    return cas

def parse_ents(cas: Cas):
    tmp_ents = []
    for ent in cas.select(T_RWSE):
        tmp_ents.append(
            {
                "start": ent.begin,
                "end": ent.end,
                "label": 'RWSE',
                "suggestion": ent.suggestion
            }
        )
    tmp_ents.sort(key=lambda x: (x['start'], x['end']))
    final_text = cas.sofa_string
    final_ents = []
    offset = 0
    for ent in tmp_ents:
        ent['start'] += offset
        ent['end'] += offset
        final_ents.append(ent)
        final_text = final_text[:ent['end']] + " " + ent['suggestion'] + final_text[ent['end']:]
        suggestion_length = len(ent['suggestion'])
        final_ents.append(
            {
                "start": ent['end'] + 1,
                "end": ent['end'] + suggestion_length + 1 ,
                "label": 'CORR'
            }
        )
        offset += suggestion_length + 1
    final_ents.sort(key=lambda x: (x['start'], x['end']))
    return spacy.displacy.EntityRenderer({"colors": labels_to_colors}).render_ents(final_text, final_ents, "")


def create_html():
    st.session_state[STATE_CAS] = create_cas(st.session_state[STATE_TEXT_AREA])
    html = parse_ents(st.session_state[STATE_CAS])
    st.write(html + '<br>', unsafe_allow_html=True)

def show_data_frame():
    data = []
    for annotation in st.session_state[STATE_CAS].select(T_RWSE):
        data.append((annotation.get_covered_text(), annotation.suggestion, annotation.certainty))
    if len(data) > 0:
        st.dataframe(pd.DataFrame.from_records(data, columns=['original', 'suggestion', 'certainty']),
                     hide_index=True)


st.title("RWSE Demo")
st.text_area(r"$\textsf{\large Enter a text here:}$",
             value=default_text,
             key=STATE_TEXT_AREA,
             height=200)
st.write(r"$\textsf{\normalsize View annotation results:}$")
with st.container(border=True):
    create_html()
st.write(r"$\textsf{\normalsize RWSE overview:}$")
show_data_frame()

with st.sidebar:
    st.write(r"$\textsf{\normalsize Add new confunsion sets:}$")
    prompt = st.chat_input("separate words with a ;")
    if prompt:
        text_input = prompt
        new_confusion_set = [item.strip() for item in text_input.split(";")]
        if len(new_confusion_set) > 1:
            cs_id = next(st.session_state[STATE_INFINITE_SEQUENCE])
            st.session_state[STATE_CONFUSION_SETS_MODIFIED].append((new_confusion_set, cs_id))
            st.session_state[STATE_CHECKBOX_ + str(cs_id)] = True
            reset_confusion_sets()
        else:
            st.write(f"Input is not valid.")
    st.divider()
    for confusion_set, cs_id in st.session_state[STATE_CONFUSION_SETS_MODIFIED]:
        st.checkbox('{' +', '.join(confusion_set) +'}', key=STATE_CHECKBOX_ + str(cs_id),
                    on_change=reset_confusion_sets)
