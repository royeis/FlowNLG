import sacrebleu


def encode_triple_string(triple, flipped=False):
    comps = triple.split(' | ')
    pred_delim = ' <P> ' if not flipped else ' <P> * '
    return '<S> ' + comps[0] + pred_delim + comps[1] + ' <O> ' + comps[2]


def encode_triple_object(triple):
    pred_delim = ' <P> ' if not triple.flipped else ' <P> * '
    return '<S> ' + triple.first + pred_delim + triple.predicate + ' <O> ' + triple.second


def string_encode_plan(plan):
    enc = ''
    for sent in plan:
        enc += '<sentence> '
        for triple in sent:
            enc += encode_triple_string(triple[0], flipped=triple[1]) + ' '  
    return enc.rstrip()


def realize_all_plans(plans, tokenizer, realizer):
    plans = [string_encode_plan(p) for p in plans]
    encs = tokenizer(plans, return_tensors='pt', padding=True)
    encs.to(realizer.device)
    gen_tokens = realizer.generate(**encs, max_length=512)
    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


def best_generation_index(generations, reference):
    bleus = [sacrebleu.corpus_bleu(g, [[reference]]).score for g in generations]
    max_score = 0.0
    max_i = -1
    for i, score in enumerate(bleus):
        if score > max_score:
            max_score = score
            max_i = i

    return max_i


def get_best_plan(plans, reference, tokenizer, realizer):
    generations = realize_all_plans(plans, tokenizer, realizer)
    best_i = best_generation_index(generations, reference)
    return plans[best_i]