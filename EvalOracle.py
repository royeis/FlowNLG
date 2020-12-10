import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sacrebleu

from DatasetReader import run_parser
from Preprocess import encode_triple_object


def encode_plan(flowtripleset):
    enc = ''
    for sent in flowtripleset:
        enc += '<sentence> '
        for triple in sent:
            enc += encode_triple_object(triple) + ' '
    return enc.rstrip()


def prepare_entry(entry, tokenizer, realizer):
    refs = []
    plans = []
    for lex in entry.lexEntries:
        refs.append(lex.text)
        plans.append(encode_plan(lex.flowtripleset))

    encs = tokenizer(plans, return_tensors='pt', padding=True)
    encs.to(realizer.device)
    gen_tokens = realizer.generate(**encs, max_length=512)
    gens = tokenizer.batch_decode(gen_tokens)
    return gens, refs


def entry_bleu(gens, refs):
    refs_ = [[r] for r in refs]
    bleus = [sacrebleu.corpus_bleu(g, refs_).score for g in gens]
    return max(bleus)


def subset_bleus(gens, refs, ids):
    gens_entries = []
    refs_entries = []

    cur_gens = []
    cur_refs = []
    for i in range(len(gens) - 1):
        cur_gens.append(gens[i])
        cur_refs.append(refs[i])
        if ids[i] != ids[i + 1]:
            gens_entries.append(cur_gens)
            refs_entries.append(cur_refs)
            cur_gens = []
            cur_refs = []

    # deal with last lex entry
    cur_gens.append(gens[-1])
    cur_refs.append(refs[-1])
    gens_entries.append(cur_gens)
    refs_entries.append(cur_refs)

    bleus = []
    for entry_gens, entry_refs in zip(gens_entries, refs_entries):
        bleus.append(entry_bleu(entry_gens, entry_refs))

    return bleus


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'utilizing device: {device}')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    realizer = T5ForConditionalGeneration.from_pretrained('realizer')
    realizer.to(device)
    realizer.eval()

    test_entries_seen = run_parser('data/test_seen')
    test_entries_unseen = run_parser('data/test_unseen')

    seen_gens = []
    seen_refs = []
    seen_entry_ids = []
    print('========= seen categories partition ==========')
    for i, entry in enumerate(test_entries_seen):
        if i % 50 == 0:
            print(f'generating text from entry {i + 1} / {len(test_entries_seen)}')
        gens, refs = prepare_entry(entry, tokenizer, realizer)
        seen_gens.extend(gens)
        seen_refs.extend(refs)
        seen_entry_ids.extend([i] * len(gens))

    seen_bleus = subset_bleus(seen_gens, seen_refs, seen_entry_ids)

    unseen_gens = []
    unseen_refs = []
    unseen_entry_ids = []
    print('========= unseen categories partition ==========')
    for i, entry in enumerate(test_entries_unseen):
        if i % 50 == 0:
            print(f'generating text from entry {i + 1} / {len(test_entries_unseen)}')
        gens, refs = prepare_entry(entry, tokenizer, realizer)
        unseen_gens.extend(gens)
        unseen_refs.extend(refs)
        unseen_entry_ids.extend([i] * len(gens))

    unseen_bleus = subset_bleus(unseen_gens, unseen_refs, unseen_entry_ids)

    all_bleus = seen_bleus + unseen_bleus

    print('BLEU score on seen categories partition: {:.4f}'.format(sum(seen_bleus) / len(seen_bleus)))
    print('BLEU score on unseen categories partition: {:.4f}'.format(sum(unseen_bleus) / len(unseen_bleus)))
    print('BLEU score on complete test set: {:.4f}'.format(sum(all_bleus) / len(all_bleus)))

