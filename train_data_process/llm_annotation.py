import os
import csv
import json
import pathlib
import httpx
import re
import time
from openai import AzureOpenAI
import argparse, random, itertools

parser = argparse.ArgumentParser()
parser.add_argument('--temp', type=float, default=0.1,
                    help='Sampling temperature for the chat-completion call')
parser.add_argument('--no_defs', action='store_true',
                    help='DO NOT pretend one-line category definitions')
parser.add_argument('--fewshot', default=None,
                    help='Path to a JSON file with few-shot examples (omit for zero-shot)')
parser.add_argument('--no_fewshot', action='store_true',
                    help='Force zero-shot even if --fewshot is given')
parser.add_argument('--shuffle', action='store_true',
                    help='Shuffle the order of sentences inside each 50-row batch')
args = parser.parse_args()

os.environ["NO_PROXY"] = "openai4reha.privatelink.openai.azure.com, xx.xx.xx.xx"

http=httpx.Client(verify=False)
dns="https://openai4reha.privatelink.openai.azure.com"
ip="https://xx.xx.xx.xx" # confidential issue
client= AzureOpenAI(api_key="xxxxxxx", api_version="2024-02-01", azure_endpoint=ip, http_client=http) # # confidential issue

FEWSHOT = []
if args.fewshot and not args.no_fewshot:
    fp = pathlib.Path(args.fewshot)
    FEWSHOT = json.loads(fp.read_text(encoding='utf-8'))

CATEGORIES = [
    "B1300 Energy level",
    "B140 Attention functions",
    "B152 Emotional functions",
    "B440 Respiration functions",
    "B455 Exercise tolerance functions",
    "B530 Weight maintenance functions",
    "D450 Walking",
    "D550 Eating",
    "D840-D859 Work and employment",
    "B280 Sensations of pain",
    "B134 Sleep functions",
    "D760 Family relationships",
    "B164 Higher-level cognitive functions",
    "D465 Moving around using equipment",
    "D410 Changing basic body position",
    "B230 Hearing functions",
    "D240 Handling stress and other psychological demands",
    "None"]

DEFINITION = {"B1300 Energy level": "Mental functions that produce vigour and stamina",
              "B140 Attention functions": "Specific mental functions of focusing on an external stimulus or internal experience for the required period of time",
              "B152 Emotional functions": "Specific mental functions related to the feeling and affective components of the processes of the mind",
              "B440 Respiration functions": "Functions of inhaling air into the lungs, the exchange of gases between air and blood, and exhaling air",
              "B455 Exercise tolerance functions": "Functions related to respiratory and cardiovascular capacity as required for enduring pjysical exertion",
              "B530 Weight maintenance functions": "Functions of maintaining appropriate body weight, including weight gain during the development period",
              "D450 Walking": "Moving along a surface on foot, step by step, so that one foot is always on the ground, such as when strolling, sauntering, walking forwards, backwards, or sideways. Include: walking short or long distances; walking on different surfaces; walking on different surfaces; walking around obstacles",
              "D550 Eating": "Carrying out the coordinated tasks and actions of eating food that has been served, bringing it to the mouth and consuming it in culturally acceptable ways, cutting or breaking food into pieces, opening bottles and cans, using eating implements, having meals, feasting or dining. Exclude: ingestion functions (chewing, swallowing, etc.), appetite",
              "D840-D859 Work and employment": "apprenticeship (work preparation); acquiring, keeping and terminating a job; remunerative employment; non-remunerative employment",
              "B280 Sensations of pain": "Sensation of unpleasant feeling indicating potential or actual damage to some body structure",
              "B134 Sleep functions": "General mental functions of periodic, reversible and selective physical and mental disengagement from one's immediate environment accompanied by characteristic physiological changes",
              "D760 Family relationships": "Creating and maintaining kinship relationships, such as with members of the nuclear family, extended family, foster and adopted family and step-relationships, more distant relationships such as second cousins, or legal guardians",
              "B164 Higher-level cognitive functions": "Specific mental functions especially dependent on the frontal lobes of the brain, including complex goal-directed behaviours such as decision-making, abstract thinking, planning and carrying out plans, mental flexibility, and deciding which behaviours are appropriate under what circumstances; often called executive functions",
              "D465 Moving around using equipment": "Moving the whole body from place to place, on any surface or space, by using specific devices designed to facilitate moving or create other ways of moving around, such as with skates, skis, scuba equipment, swim fins, or moving down the street in a wheelchair or a walker",
              "D410 Changing basic body position": "Getting into and out of a body position and moving from one location to another, such as rolling from one side to the other, sitting, standing, getting up out of a chair to lie down on a bed, and getting into and out of positions of kneeling or squatting",
              "B230 Hearing functions": "Sensory functions relating to sensing the presence of sounds and discriminating the location, pitch, loudness and quality of sounds",
              "D240 Handling stress and other psychological demands": "Carrying out simple or complex and coordinated actions to manage and control the psychological demands required to carry out tasks demanding significant responsibilities and involving stress, distraction, or crises, such as taking exams, driving a vehicle during heavy traffic, putting on clothes when hurried by parents, finishing a task within a time-limit or taking care of a large group of children",
              "None": "Does not belong to any of the ICF categories in the list"}



# pairs each sentence back to its index and allows multi-label
def build_prompt(sentences, detailed_defs=True, fewshot=None):

    if fewshot is None:
        fewshot = []
    sys = {
        "role": "system",
        "content": (
            "You are an annotationn assistant.\n"
            "You will receive sentences from a Dutch clinical note. The sentences are already split in a list. "
            "For each sentence in the list you should choose zero, one, or more ICF categories. "
            "For each sentence, if the category 'None' is chosen, no other category should be added. "
            "Return a JSON array of objects."
        )
    }

    if detailed_defs:
        defs = "\n".join(f"- **{cat}**: {DEFINITION[cat]}" for cat in CATEGORIES)
    else:
        defs = "Categories: " + ", ".join(CATEGORIES)

    example_txt = ""
    if fewshot:
        example_txt = "### Examples (already annotated):\n" + "\n".join(
        f"- **Sentence**: {ex['sentence']}\n"
        f"  **Categories**: {', '.join(ex['categories'])}"
        for ex in FEWSHOT
        ) + "\n\n"

    user = {
        "role": "user",
        "content": (
            f"{defs}\n\n"
            f"{examples_txt}"
            "### Sentences:\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
            + "\n\n"
            "### Output format:\n"
            'Return a JSON array of objects, each with:\n'
            '  {\n'
            '    "sentence_index": <1-based index>,\n'
            '    "categories": [<0 or more of the exact category names>]\n'
            '  }\n'
        )
    }

    return [sys, user]


def main(
    input_csv: str = "newcats_sentences_2023_shuf.csv",
    output_json: str = "annotated_newcats_sentences_2023.json",
    model: str = "gpt-4o",
    detailed_defs:bool= False,
    batch_size: int = 50
):
    results = []
    buffer = []
    global_idx = 0


    with open(input_csv, newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin, delimiter=",")
        header = next(reader, None)
        for note_id, sent_idx, sentence in reader:
            buffer.append((note_id, int(sent_idx), sentence))
            # once batch_size is hit, process
            if len(buffer) >= batch_size:
                _flush(buffer, results, model, detailed_defs)
                buffer.clear()

    if buffer:
        _flush(buffer, results, model, detailed_defs)

    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} annotated sentences to {output_json}")

FLUSH_COUNT = 0

def _flush(buffer, results, model, detailed_defs):
    '''
    Send buffer of (note_id, idx, sentence) to GPT,
    parse the JSON and appends to results.
    '''

    global FLUSH_COUNT
    FLUSH_COUNT += 1
    print(f"[{time.strftime('%H:%M:%S')}] processsed {FLUSH_COUNT*len(buffer):,} sentences ...")

    # extract just the sentences for the prompt
    sentences = [t[2] for t in buffer]
    msgs = build_prompt(
            sentences,
            detailed_defs = (not args.no_defs),
            fewshot = FEWSHOT
    )

    resp = client.chat.completions.create(
        model = model,
        messages = msgs,
        temperature = args.temp
    )
    raw = resp.choices[0].message.content or ""
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```$", "", raw).strip()
    match = re.search(r"\[[\s\S]*\]\s*$", raw)
    if not match:
        print("Could not find a JSON array in the model output:")
        print(raw[:800])
        raise ValueError("No JSON array found")
    json_str = match.group(0)

    try:
        annos = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Falied to parse JSON (after extraction):", e)
        print("JSON substring was:\n", json_str[:800])
        raise

    for a in annos:
        idx = a["sentence_index"] - 1
        note_id, sent_idx, sent_text = buffer[idx]
        results.append({"note_id": note_id,
                        "sentence_index": sent_idx,
                        "sentence": sent_text,
                        "categories": a.get("categories", [])})

if __name__ == "__main__":
    main(detailed_defs=True)
