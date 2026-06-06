#!/usr/bin/env python3
import os
import json
import sys


GEC_INSTR = "Correct the orthographic, grammatical and other errors in this {lang} text segment"
DIFF_ID_INSTR = "Identify the language learner level (A1/A2/B1/B2/C1/C2) of the author of this {lang} text segment"


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


def file_to_base_folder(filename):
    return '.'.join(filename.split('.')[:-1])


def prep_out_folder(filename):
    folder_name = file_to_base_folder(filename)

    if os.path.exists(folder_name):
        raise Exception(f"Output folder '{folder_name}' already exists, don't want to overwrite.")
    else:
        os.makedirs(folder_name)

    return folder_name


def file_to_idx_name(folder, idx):
    return f"{folder}/{idx}.jsonl"


def jsonl_to_multiple_lines():
    num_threads = int(sys.argv[2])
    in_filename = sys.argv[3]
    out_folder = prep_out_folder(in_filename)

    out_fhs = [open(file_to_idx_name(out_folder, i), 'w')
               for i in range(num_threads)]

    with open(in_filename, 'r') as fh_in:
        for ii, line in enumerate(fh_in):
            out_fhs[ii % num_threads].write(line)


########################################################################
# MultiGEC
########################################################################

def multigec_read_header(line):
    fields = line.split(' ')

    err_msg = f"Tried to read MultiGEC header, but got unexpected input: {line}"

    assert line.startswith("### essay_id = "), err_msg

    if len(fields) == 4:
        result = None
    elif len(fields) == 5:
        if fields[4].startswith('(') and fields[4].endswith(')'):
            result = fields[4].strip('()')
        else:
            result = None
    else:
        raise Exception(err_msg)

    return result


def multigec_read_one(filename):
    with open(filename, 'r') as fh_in:
        result = []

        header = None
        raw_buf = []

        for raw_line in fh_in:
            line = raw_line.strip()

            if line.startswith("### essay_id = "):
                assert header is None

                header = multigec_read_header(line)
            elif line.strip() == '':
                if header is not None and len(raw_buf) > 0:
                    result.append((header, raw_buf))

                header = None
                raw_buf = []

            else:
                raw_buf.append(line)

        return result

def get_lang_from_filename(flname):
    result = None

    for cand_lang in "English Estonian German Latvian Russian Swedish Ukrainian".split(" "):
        if cand_lang.lower() in flname.lower():
            assert result is None
            result = cand_lang

    return result


def do_instr(instr, inp, outp):
    print(json.dumps({'instruct': instr, 'input': inp, 'output': outp}))


def multigec_to_instructions():
    orig_data = multigec_read_one(sys.argv[2])
    ref_data = multigec_read_one(sys.argv[3])

    file_lang = get_lang_from_filename(sys.argv[2])
    langx = get_lang_from_filename(sys.argv[3])
    assert file_lang == langx

    assert len(orig_data) == len(ref_data)

    for entry_orig, entry_ref in zip(orig_data, ref_data):
        assert entry_orig[0] == entry_ref[0]

        do_instr(GEC_INSTR.format(lang=file_lang), ' '.join(entry_orig[1]), ' '.join(entry_ref[1]))

        if entry_orig[0] is not None:
            do_instr(DIFF_ID_INSTR.format(lang=file_lang), ' '.join(entry_orig[1]), entry_orig[0])
            do_instr(DIFF_ID_INSTR.format(lang=file_lang), ' '.join(entry_ref[1]), entry_orig[0])


def est_gecde_to_instructions():
    pass

    
def say_no_to_global_variables():
    cmd = sys.argv[1]

    if cmd == 'jsonl':
        jsonl_to_multiple_lines()
    elif cmd == 'multigec':
        multigec_to_instructions()
    elif cmd == 'estgecde':
        est_gecde_to_instructions()

if __name__ == '__main__':
    say_no_to_global_variables()



    """
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
    """
