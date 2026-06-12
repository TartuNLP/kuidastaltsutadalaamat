#!/usr/bin/env python3
import os
import json
import sys
from random import shuffle


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
        elif strip_line == "},":
            buf += "}"

            entr = json.loads(buf)

            yield entr


def file_to_base_folder(filename, num_threads):
    return '.'.join(filename.split('.')[:-1]) + "-" + str(num_threads)


def prep_out_folder(filename, num_threads):
    folder_name = file_to_base_folder(filename, num_threads)

    if os.path.exists(folder_name):
        raise Exception(f"Output folder '{folder_name}' already exists, don't want to overwrite.")
    else:
        os.makedirs(folder_name)

    return folder_name


def file_to_idx_name(folder, idx):
    return f"{folder}/{idx:03}.jsonl"


def load_aux_buf(in_filename, batch_size):
    result = []

    # load first batch_size lines from in_filename
    with open(in_filename, 'r') as fh_in:
        for ii, line in enumerate(fh_in):
            if ii == batch_size:
                break
            result.append(line)
        return result


def jsonl_to_multiple_files(num_threads, batch_size, in_filename):
    assert batch_size % num_threads == 0, "batch size must be divisible by number of threads"

    out_folder = prep_out_folder(in_filename, num_threads)

    aux_buf = load_aux_buf(in_filename, batch_size)

    out_fhs = [open(file_to_idx_name(out_folder, i), 'w')
               for i in range(num_threads)]

    with open(in_filename, 'r') as fh_in:
        for ii, line in enumerate(fh_in):
            if ii % num_threads == 0:
                shuffle(out_fhs)

            out_fhs[ii % num_threads].write(line)

        #to ensure same size of chunks
        ii += 1
        while ii % batch_size != 0:
            out_fhs[ii % num_threads].write(aux_buf.pop(0))
            ii += 1

    
def say_no_to_global_variables():
    num_threads = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    in_filename = sys.argv[3]
    jsonl_to_multiple_files(num_threads, batch_size, in_filename)

if __name__ == '__main__':
    say_no_to_global_variables()
