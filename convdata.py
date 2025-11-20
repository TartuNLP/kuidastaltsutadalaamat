#!/usr/bin/env python3
import json
import sys

def parse_lang(lang_code, syn):
    dct = {
        'ar': 'Arabic',
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'et': 'Estonian',
        'fi': 'Finnish',
        'fr': 'French',
        'lt': 'Lithuanian',
        'lv': 'Latvian',
        'ru': 'Russian',
        'sv': 'Swedish',
        'uk': 'Ukrainian',
        'zh': 'Chinese',
        'pl': 'Polish'
    }

    result = dct[lang_code] + (", synth" if syn else "")
    return result


def parse_langs(raw_lp, syn):
    assert "_" in raw_lp

    tgt_lang_code, src_lang_code = raw_lp.split("_")

    return parse_lang(tgt_lang_code, syn), parse_lang(src_lang_code, syn)


def gen_out_line(fh_out, src_segm, tgt_segm, src_lang, tgt_lang, task="translate", comet_sc=None):
    data = {
        "src_segm": src_segm,
        "src_lang": src_lang,
        "tgt_segm": tgt_segm,
        "tgt_lang": tgt_lang,
        "task": task
    }

    if comet_sc is None or comet_sc > 0.85:
        fh_out.write(json.dumps(data))
        fh_out.write("\n")


def neurotolge_json_to_jsonl(input_file):
    assert input_file[5] == '/'
    assert input_file.endswith('.json')

    lp_raw, filename = input_file.split('/')
    output_file = f"{lp_raw}-{filename}l"

    is_synth = "synthetic" in filename
    lang_out, lang_in = parse_langs(lp_raw, is_synth)

    with open(input_file, 'r') as fh_in, open(output_file, 'w') as fh_out:
        raw_data = json.load(fh_in)

        assert len(raw_data.keys()) == 1

        k = list(raw_data.keys())[0]
        raw_list = raw_data[k]

        for entry in raw_list:
            gen_out_line(fh_out, entry['mt'], entry['src'], lang_in, lang_out, task="translate", comet_sc=entry['COMET'])

            if not is_synth:
                gen_out_line(fh_out, entry['src'], entry['mt'], lang_out, lang_in, task="translate", comet_sc=entry['COMET'])


def is_hi(l):
    if l.lower() in { 'et', 'en', 'lv', 'ru', 'no', 'fi', 'lt', 'fr', 'de', 'sv',
                      'est', 'eng', 'lvs', 'lav', 'rus', 'nor', 'fin', 'lit', 'fra', 'deu', 'swe' }:
        return True

    for lk in ['english', 'estonian', 'russian', 'latvian', 'finnish', 'lithuanian',
               'swedish', 'norwegian', 'french', 'german', 'swedish']:
        if lk in l.lower():
            return True

    return False


def iter_stdin_json_items(fh):
    buf = None
    for raw_line in fh:
        strip_line = raw_line.strip()
        if strip_line == "{":
            buf = raw_line
        elif "\": \"" in strip_line:
            buf += raw_line
        elif strip_line == "}":
            buf += raw_line

            entr = json.loads(buf)

            yield entr


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # convert Neurotolge json files to jsonl for training
        all_data = []

        for input_file in sys.argv[1:]:
            print(f"Processing {input_file}")
            neurotolge_json_to_jsonl(input_file)
    else:
        # convert one Smugri json file to jsonl for training
        #flname = sys.stdin.readline().strip()
        #with open(flname, 'r') as fh_i:
        for entryy in iter_stdin_json_items(sys.stdin):
            if not(is_hi(entryy['src_lang']) and is_hi(entryy['tgt_lang'])):
                gen_out_line(sys.stdout,
                             entryy['src_segm'],
                             entryy['tgt_segm'],
                             entryy['src_lang'],
                             entryy['tgt_lang'],
                             entryy['task'],
                             comet_sc=None)

