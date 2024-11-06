import pandas as pd
import torch
import esm
from os.path import join
from util import BASE_DIR
import gc


def calcualte_esm1b_vectors(protein_list):
    #creating model input:
    df_protein = preprocess_proteins(protein_list)
    model_input = [(df_protein["ID"][ind], df_protein["model_input"][ind]) for ind in df_protein.index]
    seqs = [model_input[i][1] for i in range(len(model_input))]
    #loading ESM-1b model:
    print(".....2(a) Loading ESM-1b model.")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location = join(BASE_DIR, "data", "esm1b_t33_650M_UR50S.pt"))
    batch_converter = alphabet.get_batch_converter()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    
    #Calculate ESM-1b representations
    print(".....2(b) Calculating protein representations.")
    df_protein["protein rep"] = ""

    for ind in df_protein.index:
        batch_labels, batch_strs, batch_tokens = batch_converter([(df_protein["ID"][ind], df_protein["model_input"][ind])])
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
        results = {layer: t.to(device="cpu") for layer, t in results["representations"].items()}
        df_protein["protein rep"][ind] = results[33][0, 1 : len(df_protein["model_input"][ind]) + 1].mean(0).numpy()

    del model, batch_tokens, results, alphabet, batch_converter
    torch.cuda.empty_cache()
    gc.collect()
    return(df_protein)


def preprocess_proteins(protein_list):
    df_protein = pd.DataFrame(data = {"amino acid sequence" : list(set(protein_list))})
    df_protein["ID"] = ["protein_" + str(ind) for ind in df_protein.index]
    #if length of sequence is longer than 1020 amino acids, we crop it:
    df_protein["model_input"] = [seq[:1022].replace("\r", "").replace("\r", "").replace("\n", "") for seq in df_protein["amino acid sequence"]]
    return(df_protein)


aa = set("abcdefghiklmnpqrstxvwyzv".upper())

def validate_protein(seq, alphabet=aa):
    "Checks that a sequence only contains values from an alphabet"
    leftover = set(seq.upper()) - alphabet
    return not leftover
