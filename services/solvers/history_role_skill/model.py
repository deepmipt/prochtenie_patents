from collections import defaultdict
from io import StringIO

from udapi.block.read.conllu import Conllu


def count_entities(parses, labels, entity_type=None, use_lemmas=False):
    answer = defaultdict(int)
    for sent_parse, sent_labels in zip(parses, labels):
        curr_entity = None
        for node, label in zip(sent_parse, sent_labels):
            if curr_entity is not None:
                if label[2:] == curr_entity_type:
                    curr_entity += " " + (node.lemma if use_lemmas else node.form)
                    continue
                else:
                    answer[curr_entity] += 1
                    curr_entity = None
            if label[0] == "B" and (entity_type is None or label[2:] == entity_type):
                curr_entity = node.lemma if use_lemmas else node.form
                curr_entity_type = label[2:]
        if curr_entity is not None:
            answer[curr_entity] += 1
    return answer


def detect_roles(parses, ner_labels, sents=None, context_length=1, min_person_count=1, return_sentence_indexes=False):
    prelim_answer = defaultdict(list)
    if len(parses) > 0 and isinstance(parses[0], list):
        parses = ["\n".join("\t".join(elem) for elem in parse) for parse in parses]
    if len(parses) > 0 and isinstance(parses[0], str):
        parses = [Conllu(filehandle=StringIO(parse)).read_tree().descendants for parse in parses]
    if sents is None:
        sents = [" ".join(node.form for node in sent_parse) for sent_parse in parses]
    entity_counts = count_entities(parses, ner_labels, entity_type="PER", use_lemmas=True)
    for r, (sent_parse, curr_labels) in enumerate(zip(parses, ner_labels)):
        " ".join(node.form for node in sent_parse)
        for i, (node, label) in enumerate(zip(sent_parse, curr_labels)):
            if label == "B-PER":
                head_node = node
                while head_node.deprel in ["conj", "appos"]:
                    head_node = head_node.parent
                if "nsubj" not in head_node.deprel:
                    continue
                person = sent_parse[i].lemma
                for j, other_node in enumerate(sent_parse[i + 1 :], i + 1):
                    if curr_labels[j] != "I-PER":
                        break
                    person += " " + other_node.lemma
                prelim_answer[person.lower()].append((r, r + context_length))
    normalized_persona = []
    for person, data in sorted(entity_counts.items(), key=lambda x: -len(x[0])):
        for other in normalized_persona:
            if (person.lower() + " ") in other.lower() or (" " + person.lower()) in other.lower():
                entity_counts[other.lower()] += entity_counts[person.lower()]
                prelim_answer[other] += prelim_answer[person]
                break
        else:
            normalized_persona.append(person)
    answer = {
        person: sorted(prelim_answer[person])
        for person in normalized_persona
        if entity_counts[person.lower()] >= min_person_count and len(prelim_answer[person]) > 0
    }
    if not return_sentence_indexes:
        for person, indexes in answer.items():
            answer[person] = [sents[start:end] for start, end in indexes]
    return answer
