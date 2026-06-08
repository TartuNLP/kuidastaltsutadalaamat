#!/usr/bin/env python3
import json
import sys
import re

from collections import namedtuple

from aux import log

IndiCorr = namedtuple('IndiCorr', 'err_class correction')


INSTR_GEC = "Correct the orthographic, grammatical and other errors in this {synth}{lang} text segment"

INSTR_GECSNT = ("Correct the orthographic, grammatical and other errors in this {synth}Estonian sentence, "
                "given preceding context sentences")
INPUT_GECSNT = "Input context: {context}\nInput sentence: {input_sent}"

INSTR_DIFF_ID = "Identify the language learner level (A1/A2/B1/B2/C1/C2) of the author of this {synth}{lang} text segment"


SNTPAIR_INPUT_WDIFF = "Input sentence: {input_sent}\nCorrected sentence: {corr_sent}\nInitial diff: {diff}"
INSTR_TOK_GRP_CORR_CLASS = ("Group and re-classify individual corrections "
                            "for the tokenized Estonian sentence, its corrected version and an initial diff between them")


SNTPAIR_INPUT_NOCTX = "Input sentence: {input_sent}\nCorrected sentence: {corr_sent}"
SNTPAIR_INPUT_WCTX = "Input context: {context}\n" + SNTPAIR_INPUT_NOCTX
SNTPAIR_INPUT_CORRS = "\nCorrections: {corrs}"

INSTR_CORRS_LIST = ("List all the individual corrections "
                    "in the given Estonian sentence{context} and its corrected version")
INSTR_CORRS_CLASS = ("Classify the provided individual corrections "
                     "in the given Estonian sentence{context} and its corrected version")
INSTR_CORRS_LIST_CLASS = ("List and classify all the individual corrections "
                          "in the given Estonian sentence{context} and its corrected version")

SNTPAIR_INPUT_FOCUS_CORR = "\nThe correction to process: {corr}"

INSTR_CORR_CLASS = ("Classify the following single correction "
                    "in the given Estonian sentence its corrected version{context} and list of all corrections")
INSTR_CORR_EXPLAIN = ("Provide a {complexity} justification for the following classified single correction "
                      "in the given Estonian sentence, its corrected version{context} and list of all corrections")
INSTR_CORR_CLASS_AND_EXPLAIN = ("Classify and provide a {complexity} justification for the following single correction "
                                "in the given Estonian sentence, its corrected version{context} and list of all corrections")

CORR_SIMPLE = "short and simple"
CORR_COMPL = "comprehensive and detailed"

OUTPUT_EXPL_CLASS = "Correction class: {class_name}\nJustification: {justif}"

def do_instr(instr, inp, outp):
    print(json.dumps({'instruct': instr, 'input': inp, 'output': outp}))

########################################################################
# MultiGEC
########################################################################

def multigec_read_header(line):
    fields = line.split(' ')

    err_msg = f"Tried to read MultiGEC header, but got unexpected input: {line}"

    assert line.startswith("### essay_id = "), err_msg

    if len(fields) == 4:
        result = '-'
    elif len(fields) == 5:
        if fields[4].startswith('(') and fields[4].endswith(')'):
            result = fields[4].strip('()')
        else:
            result = '-'
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


def multigec_to_instructions():
    orig_data = multigec_read_one(sys.argv[2])
    ref_data = multigec_read_one(sys.argv[3])

    file_lang = get_lang_from_filename(sys.argv[2])
    langx = get_lang_from_filename(sys.argv[3])
    assert file_lang == langx

    assert len(orig_data) == len(ref_data)

    log(f"Successfully read {len(orig_data)} pairs")

    for entry_orig, entry_ref in zip(orig_data, ref_data):
        assert entry_orig[0] == entry_ref[0]

        do_instr(INSTR_GEC.format(lang=file_lang, synth=""), ' '.join(entry_orig[1]), ' '.join(entry_ref[1]))

        if entry_orig[0] != '-':
            do_instr(INSTR_DIFF_ID.format(lang=file_lang, synth=""), ' '.join(entry_orig[1]), entry_orig[0])
            do_instr(INSTR_DIFF_ID.format(lang=file_lang, synth=""), ' '.join(entry_ref[1]), entry_orig[0])

########################################################################
# Estonian GEC, GED and GEE
########################################################################

def est_gec_to_instr(entry, is_synth):
    if entry['filter_lvl'] == 'unfiltered':
        return

    if 'input_essay' in entry:
        do_instr(
            INSTR_GEC.format(lang="Estonian", synth=("synthetic " if is_synth else "")),
            entry['input_essay'],
            entry['output_essay'])

        do_instr(
            INSTR_DIFF_ID.format(lang="Estonian", synth=("synthetic " if is_synth else "")),
            entry['input_essay'],
            entry['prof_level'])

        do_instr(
            INSTR_DIFF_ID.format(lang="Estonian", synth=("synthetic " if is_synth else "")),
            entry['output_essay'],
            entry['prof_level'])

    elif 'input_sent' in entry:
        do_instr(
            INSTR_GECSNT.format(synth=("synthetic " if is_synth else "")),
            INPUT_GECSNT.format(**entry),
            entry['output_sent'])


def est_ged_to_instr(entry):
    do_instr(
        INSTR_TOK_GRP_CORR_CLASS,
        SNTPAIR_INPUT_WDIFF.format(input_sent=entry['input_sent'], corr_sent=entry['output_sent'], diff=entry['erinevused']),
        entry['parandused'])


def _parse_corrections(raw_corrections):
    result = []

    for raw_correction in raw_corrections.split('\n'):
        m = re.fullmatch(r'^([0-9]+). ([^:]+): (.+ -> .+)$', raw_correction)
        result.append(IndiCorr(m.group(2), m.group(3)))

    return result


def _filter_corrections(raw_corrections, cl=False, corr=False):
    output_fields = []

    if cl:
        output_fields.append('err_class')
    if corr:
        output_fields.append('correction')

    corr_list = _parse_corrections(raw_corrections)

    flt_list = [": ".join([ic._asdict()[ky] for ky in output_fields]) for ic in corr_list]

    return "\n".join(flt_list)


def _get_corr_key(entry):
    result = None

    for k in entry.keys():
        if k.startswith('selgitus_'):
            assert result is None, "selgitus_ key already found, duplicates not allowed"
            result = k

    assert result is not None, "no selgitus_ key found"

    return result

def est_gee_to_instr(entry):
    if entry['kontekst'] == "":
        ctx_snt = ""
        tmpl = SNTPAIR_INPUT_NOCTX
    else:
        ctx_snt = " with preceding context sentences"
        tmpl = SNTPAIR_INPUT_WCTX

    # list corrections (w/wo ctx):
    #   (kontekst,) algne_lause, parandatud_lause --> parandused (no classes)
    do_instr(
        INSTR_CORRS_LIST.format(context=ctx_snt),
        tmpl.format(input_sent=entry['algne_lause'],
                    corr_sent=entry['parandatud_lause'],
                    context=entry['kontekst']),
        _filter_corrections(entry['parandused'], corr=True))

    # list and classify corrections (w/wo ctx):
    #   (kontekst,) algne_lause, parandatud_lause --> parandused
    do_instr(
        INSTR_CORRS_LIST_CLASS.format(context=ctx_snt),
        tmpl.format(input_sent=entry['algne_lause'],
                    corr_sent=entry['parandatud_lause'],
                    context=entry['kontekst']),
        _filter_corrections(entry['parandused'], cl=True, corr=True))

    # classify corrections (w/wo ctx):
    #   (kontekst,) algne_lause, parandatud_lause, parandused (no classes) --> parandused (yes classes)
    do_instr(
        INSTR_CORRS_CLASS.format(context=ctx_snt),
        tmpl.format(input_sent=entry['algne_lause'],
                    corr_sent=entry['parandatud_lause'],
                    context=entry['kontekst'])
        + SNTPAIR_INPUT_CORRS.format(corrs=_filter_corrections(entry['parandused'], corr=True)),
        _filter_corrections(entry['parandused'], cl=True, corr=True))

    #################
    corr_key = _get_corr_key(entry)
    
    input_corr_class_first_part = (
            tmpl.format(
                input_sent=entry['algne_lause'],
                corr_sent=entry['parandatud_lause'],
                context=entry['kontekst'])
            + SNTPAIR_INPUT_CORRS.format(corrs=_filter_corrections(entry['parandused'], corr=True)))

    second_part_without_class = SNTPAIR_INPUT_FOCUS_CORR.format(corr=entry[corr_key])
    second_part_with_class = SNTPAIR_INPUT_FOCUS_CORR.format(corr=entry['vealiik'] + ": " + entry[corr_key])

    # classify the following correction (nr N) (w/wo ctx):
    #   (kontekst,) algne_lause, parandatud_lause, parandused, selgitus_N --> vealiik
    do_instr(
        INSTR_CORR_CLASS.format(context=ctx_snt),
        input_corr_class_first_part + second_part_without_class,
        entry['vealiik'])

    #OUTPUT_EXPL_CLASS = "Correction class: {class_name}\nJustification: {justif}"
    # classify and explain simply/fully corr (nr N) (w/wo ctx):
    #   (kontekst,) algne_lause, parandatud_lause, parandused, selgitus_N --> vealiik, lühike/pikk
    do_instr(
        INSTR_CORR_CLASS_AND_EXPLAIN.format(context=ctx_snt, complexity=CORR_SIMPLE),
        input_corr_class_first_part + second_part_without_class,
        OUTPUT_EXPL_CLASS.format(justif=entry['lühike'], class_name=entry['vealiik']))

    do_instr(
        INSTR_CORR_CLASS_AND_EXPLAIN.format(context=ctx_snt, complexity=CORR_COMPL),
        input_corr_class_first_part + second_part_without_class,
        OUTPUT_EXPL_CLASS.format(justif=entry['pikk'], class_name=entry['vealiik']))

    # explain simply/fully the following corr (nr N) (w/wo ctx):
    #   (kontekst,) algne_lause, parandatud_lause, parandused, selgitus_N, vealiik --> lühike/pikk
    do_instr(
        INSTR_CORR_EXPLAIN.format(context=ctx_snt, complexity=CORR_SIMPLE),
        input_corr_class_first_part + second_part_with_class,
        entry['lühike'])

    do_instr(
        INSTR_CORR_EXPLAIN.format(context=ctx_snt, complexity=CORR_COMPL),
        input_corr_class_first_part + second_part_with_class,
        entry['pikk'])


def est_gecde_to_instructions():
    filename = sys.argv[2]

    with open(filename, 'r') as fh_in:
        for raw_line in fh_in:
            entry = json.loads(raw_line)

            if "ged/" in filename:
                est_ged_to_instr(entry)
            elif "gee/" in filename:
                est_gee_to_instr(entry)
            elif "gec/" in filename:
                if "ut_l2" in filename:
                    entry['filter_lvl'] = 'itsok'
                est_gec_to_instr(entry, "synth" in filename)


def say_no_to_global_variables():
    cmd = sys.argv[1]

    if cmd == 'multigec':
        multigec_to_instructions()
    elif cmd == 'estgecde':
        est_gecde_to_instructions()
    else:
        raise Exception(f"Unknown command: {cmd}")

if __name__ == '__main__':
    say_no_to_global_variables()