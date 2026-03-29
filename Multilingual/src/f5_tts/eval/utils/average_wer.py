import sys
import numpy as np

infile=sys.argv[1]
outfile=sys.argv[2]
whole=True
fout = open(outfile, "w")
if not whole:
    fout.write("wav_res" + '\t' + 'res_wer' + '\t' + 'text_ref' + '\t' + 'text_res' + '\t' + 'res_wer_ins' + '\t' + 'res_wer_del' + '\t' + 'res_wer_sub' + '\n')
    wers = []
    inses = []
    deles = []
    subses = []
    for line in open(infile, "r").readlines():
        wav_path, wer, text_ref, text_res, inse, dele, subs = line.strip().split("\t")
            
        wers.append(float(wer))
        inses.append(float(inse))
        deles.append(float(dele))
        subses.append(float(subs))
        fout.write(line)

    wer = round(np.mean(wers)*100,3)
    subs = round(np.mean(subses)*100,3)
    dele = round(np.mean(deles)*100,3)
    inse = round(np.mean(inses)*100,3)

    subs_ratio = round(subs / wer, 3)
    dele_ratio = round(dele / wer, 3)
    inse_ratio = round(inse / wer, 3)

    fout.write(f"WER: {wer}%, SUB: {subs_ratio}%, DEL: {dele_ratio}%, INS: {inse_ratio}%\n")
    fout.close()
else:
    fout.write("wav_res" + '\t' + 'edit_distance' + '\t' + 'len_words' + '\t' + 'wer_per_sen' + '\t' + 'text_ref' + '\t' + 'text_res'  + '\n')
    scores = 0
    words = 0
    wers = []
    for line in open(infile, "r").readlines():
        wav_path, edit_distance, len_words, wer_per_sen, text_ref, text_res = line.strip().split("\t")
        scores += float(edit_distance)
        words += float(len_words)
        wers.append(float(wer_per_sen))
        fout.write(line)
    wer = round(scores / words * 100, 3)
    wer_avg = round(np.mean(wers)*100,3)
    fout.write(f"WER: {wer}%, WER_avg: {wer_avg}%\n")
    fout.close()

print(f"WER: {wer}%\nWER(average on sentences): {wer_avg}%\n")
