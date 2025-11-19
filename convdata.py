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


def gen_out_line(fh_out, src_segm, tgt_segm, src_lang, tgt_lang, task = "translate", comet = None):
    data = {
        "src_segm": src_segm,
        "src_lang": src_lang,
        "tgt_segm": tgt_segm,
        "tgt_lang": tgt_lang,
        "task": task
    }

    if comet is not None:
        data["COMET"] = comet

    fh_out.write(json.dumps(data))
    fh_out.write("\n")


def neurotolge_json_to_jsonl(input_file):
    assert input_file[5] == '/'
    assert input_file.endswith('.json')

    lp_raw, filename = input_file.split('/')
    output_file = f"{lp_raw}-{filename}.jsonl"

    is_synth = "synthetic" in filename
    lang_out, lang_in = parse_langs(lp_raw, is_synth)

    with open(input_file, 'r') as fh_in, open(output_file, 'w') as fh_out:
        raw_data = json.load(fh_in)

        assert len(raw_data.keys()) == 1

        k = list(raw_data.keys())[0]
        raw_list = raw_data[k]

        for entry in raw_list:
            gen_out_line(fh_out, entry['mt'], entry['src'], lang_in, lang_out, entry['COMET'])

            if not is_synth:
                gen_out_line(fh_out, entry['src'], entry['mt'], lang_out, lang_in, entry['COMET'])


def is_hi(l):
    if l.lower() in { 'et', 'en', 'lv', 'ru', 'no', 'fi', 'lt', 'fr', 'de', 'sv',
                      'est', 'eng', 'lvs', 'lav', 'rus', 'nor', 'fin', 'lit', 'fra', 'deu', 'swe' }:
        return True

    for lk in ['english', 'estonian', 'russian', 'latvian', 'finnish', 'lithuanian',
               'swedish', 'norwegian', 'french', 'german', 'swedish']:
        if lk in l.lower():
            return True

    return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # convert Neurotolge json files to jsonl for training
        all_data = []

        for input_file in sys.argv[1:]:
            print(f"Processing {input_file}")
            neurotolge_json_to_jsonl(input_file)
    else:
        # convert one Smugri json file to jsonl for training
        smugri_data = json.load(sys.stdin)

        for entry in smugri_data:
            if is_hi(entry['src_lang']) and is_hi(entry['tgt_lang']):
                comet = 42
            else:
                comet = None

            gen_out_line(sys.stdout, **entry, comet=comet)

