from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import xml.etree.ElementTree as ET
import re
from xml.dom import minidom

from Preprocess import get_best_plan


# a generator for flat flow hints permutations given a flat list of triplets
def triplelist_to_perms(triple_list):
    if len(triple_list) == 0:
        yield [], []
    else:
        triple = triple_list[0]
        comps = triple.split(' | ')
        reversed_triple = comps[2] + ' | ' + comps[1] + ' | ' + comps[0]

        for (perm, flips) in triplelist_to_perms(triple_list[1:]):
            yield [triple] + perm, [False] + flips
            yield [reversed_triple] + perm, [True] + flips


# a generator for all information flow hints induced plans.
def flow_hints_permutations(plan):
    # param plan: list[list[String]]

    sent_lens = [len(sent) for sent in plan]
    flat_tripleset = [triple for sent in plan for triple in sent]

    for (flat_perm, flips) in triplelist_to_perms(flat_tripleset):
        plan = []
        count = 0
        for length in sent_lens:
            sent_triplets = list(zip(flat_perm[count:count + length], flips[count:count + length]))
            plan.append(sent_triplets)
            count += length

        yield plan


def xml_lex_entry_to_plan(entry):
    sents = entry.find('sortedtripleset').findall('sentence')
    return [[t.text for t in sent.findall('striple')] for sent in sents]


def refine_xml(in_file, out_file, tokenizer, realizer):
    tree = ET.parse(in_file)
    root = tree.getroot()
    entries = root.find('entries')

    for j, entry in enumerate(entries):
        if j % 10 == 0:
            print(f'computing best plan for entry: {j+1} / {len(entries)}')
        lex_entries = entry.findall('lex')
        for lex_entry in lex_entries:
            ref = lex_entry.find('text').text

            plan = xml_lex_entry_to_plan(lex_entry)
            flow_hints_plans = list(flow_hints_permutations(plan))
            original_plan = flow_hints_plans[0]
            try:
                best_plan = get_best_plan(flow_hints_plans, ref, tokenizer, realizer)
            except EOFError:
                print(f'unable to compute best plan for entry: {j}\noriginal plan: {original_plan}\nreference: {ref}')
                continue

            flow_induced_tripleset = ET.SubElement(lex_entry, 'flowinducedtripleset')
            for i, sent in enumerate(best_plan):
                flow_induced_sent = ET.SubElement(flow_induced_tripleset, 'sentence')
                flow_induced_sent.attrib = {'ID': str(i + 1)}

                for triple in sent:
                    striple = ET.SubElement(flow_induced_sent, 'ftriple')
                    striple.text = triple[0]
                    striple.attrib = {'flipped': str(triple[1])}

    rough_string = ET.tostring(tree.getroot(), encoding='utf-8', method='xml')
    rough_string = re.sub(">\n[\t]+<", '><', rough_string.decode('utf-8'))
    xml = minidom.parseString(rough_string).toprettyxml(indent="\t")

    with open(out_file, 'wb+') as f:
        f.write(xml.encode('utf-8'))


def refine_dataset(in_dir, out_dir, tokenizer, realizer, overwrite=False):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    subsetdirs = filter(lambda item: not str(item).startswith('.'), os.listdir(in_dir))
    for subsetdir in subsetdirs:
        print(f'Refining {subsetdir} data subset')
        sub_dir = os.path.join(out_dir, subsetdir)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        ntriplesdirs = filter(lambda item: not str(item).startswith('.'), os.listdir(os.path.join(in_dir, subsetdir)))
        for ntriplesdir in ntriplesdirs:
            print(f'directory: {ntriplesdir}')
            sub_n_dir = os.path.join(sub_dir, ntriplesdir)
            if not os.path.exists(sub_n_dir):
                os.mkdir(sub_n_dir)

            xmlfiles = filter(lambda item: not str(item).startswith('.'), os.listdir(os.path.join(in_dir, subsetdir, ntriplesdir)))
            for xmlfile in xmlfiles:
                print(f'xml: {xmlfile}')
                input_file = os.path.join(in_dir, subsetdir, ntriplesdir, xmlfile)
                output_file = os.path.join(out_dir, subsetdir, ntriplesdir, xmlfile)
                if not os.path.exists(output_file) or overwrite:
                    refine_xml(input_file, output_file, tokenizer, realizer)


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'utilizing device: {device}')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    realizer = T5ForConditionalGeneration.from_pretrained('realizer')
    realizer.to(device)
    realizer.eval()

    refine_dataset('DeepNLG_data/v1.4/en', 'FlowNLG_data', tokenizer, realizer)