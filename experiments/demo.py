import streamlit as st
import spacy
import unicodedata

from cassis import Cas, load_typesystem
from rwse_checker.rwse import RWSE_Checker, T_RWSE, T_SENTENCE, T_TOKEN
from transformers.utils import logging

logging.set_verbosity_error()

STATE_TEXT_AREA = 'STATE_TEXT_AREA'

nlp = spacy.load("en_core_web_sm")

ts_file = 'experiments/input/TypeSystem.xml'
with open(ts_file, 'rb') as f:
    ts = load_typesystem(f)

rwse_checker = RWSE_Checker()
rwse_checker.set_confusion_sets('experiments/input/confusion_sets_modified.csv')

default_text = "My advise for you is: Do not put to much subjects, just put a few subject and make them look interesting."

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
    rwse_checker.check_cas(cas, ts)
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
    cas = create_cas(st.session_state[STATE_TEXT_AREA])
    html = parse_ents(cas)
    st.write(html + '<br>', unsafe_allow_html=True)


st.title("RWSE Demo")
st.text_area(r"$\textsf{\large Enter a text here:}$",
             value=default_text,
             key=STATE_TEXT_AREA,
             height=200)
st.write(r"$\textsf{\normalsize View annotation results:}$")
with st.container(border=True):
    create_html()


