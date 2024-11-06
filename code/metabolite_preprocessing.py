import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from os.path import join
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
warnings.filterwarnings("ignore")
from util import BASE_DIR

tokenizer = AutoTokenizer.from_pretrained(join(BASE_DIR, "data", "761d6a18cf99db371e0b43baf3e2d21b3e865a20"))
model = AutoModelForMaskedLM.from_pretrained(join(BASE_DIR, "data", "761d6a18cf99db371e0b43baf3e2d21b3e865a20"))


df_metabolites = pd.read_csv(join(BASE_DIR, "data", "all_substrates.csv"))
for ind in df_metabolites.index:
    mol = Chem.inchi.MolFromInchi(df_metabolites["ID"][ind])
    df_metabolites["Sim_FP"][ind] = Chem.RDKFingerprint(mol)


def metabolite_preprocessing_chemberta(metabolite_list):
  #removing duplicated entries and creating a pandas DataFrame with all metabolites
  df_met = pd.DataFrame(data = {"metabolite" : list(set(metabolite_list))})
  df_met["type"], df_met["ID"] = np.nan, np.nan

  #each metabolite should be either a KEGG ID, InChI string, or a SMILES:
  for ind in df_met.index:
    df_met["ID"][ind] = "metabolite_" + str(ind)
    met = df_met["metabolite"][ind]
    if type(met) != str:
      df_met["type"][ind] = "invalid"
      print(".......Metabolite string '%s' could be neither classified as a valid KEGG ID, InChI string or SMILES string" % met)
    elif is_KEGG_ID(met):
      df_met["type"][ind] = "KEGG"
    elif is_InChI(met):
      df_met["type"][ind] = "InChI"
    elif is_SMILES(met):
      df_met["type"][ind] = "SMILES"
    else:
      df_met["type"][ind] = "invalid"
      print(".......Metabolite string '%s' could be neither classified as a valid KEGG ID, InChI string or SMILES string" % met)
  df_met = calculate_chemberta_vectors(df_met)
  df_met = calculate_ecfps(df_met)
  return(df_met)


def is_KEGG_ID(met):
  #a valid KEGG ID starts with a "C" or "D" followed by a 5 digit number:
  if len(met) == 6 and met[0] in ["C", "D"]:
    try:
      int(met[1:])
      return(True)
    except: 
      pass
  return(False)

def is_SMILES(met):
  m = Chem.MolFromSmiles(met,sanitize=False)
  if m is None:
    return(False)
  else:
    try:
      Chem.SanitizeMol(m)
    except:
      print('.......Metabolite string "%s" is in SMILES format but has invalid chemistry' % met)
      return(False)
  return(True)

def is_InChI(met):
  m = Chem.inchi.MolFromInchi(met,sanitize=False)
  if m is None:
    return(False)
  else:
    try:
      Chem.SanitizeMol(m)
    except:
      print('.......Metabolite string "%s" is in InChI format but has invalid chemistry' % met)
      return(False)
  return(True)

  
def calculate_ecfps(df_met):
    df_met["successfull"] = True
    df_met["ECFP"] = ""
    df_met["metabolite_similarity_score"] = np.nan
    df_met["metabolite_identical_ID"] = np.nan
    df_met["#metabolite in training set"] = np.nan

    df_count_met = pd.read_csv(join(BASE_DIR, "data", "all_training_metabolites.csv"), sep = "\t")
    df_count_met["ID"] = df_count_met["InChI"]


    for ind in df_met.index:
        ID, met_type, met = df_met["ID"][ind], df_met["type"][ind], df_met["metabolite"][ind]
        if met_type == "invalid":
            mol = None
        elif met_type == "KEGG":
            try:
                mol = Chem.MolFromMolFile(join(BASE_DIR, "data", "mol-files",  met + ".mol"))
            except:
                print(".......Mol file for KEGG ID '%s' is not available. Try to enter InChI string or SMILES for the metabolite instead." % met)
                mol = None
        elif met_type == "InChI":
            mol = Chem.inchi.MolFromInchi(met)
        elif met_type == "SMILES":
            mol = Chem.MolFromSmiles(met)
        if mol is None:
            df_met["successfull"][ind] = False
        '''else:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToBitString()
            df_met["ECFP"][ind] = ecfp
            df_met["metabolite_similarity_score"][ind], df_met["metabolite_identical_ID"][ind] = calculate_metabolite_similarity(df_metabolites = df_metabolites,
                                                                                                     mol = mol)
            if not pd.isnull(df_met["metabolite_identical_ID"][ind]):
                df_met["#metabolite in training set"][ind] = list(df_count_met["count"].loc[df_count_met["ID"] == df_met["metabolite_identical_ID"][ind]])[0]
            else:
                df_met["#metabolite in training set"][ind] = 0'''
    return(df_met)

def getMeanRepr(smiles_data):
    mean_repr = np.zeros((smiles_data.shape[0], 767))
    for i, sequence in enumerate(smiles_data):
        inputs = tokenizer.encode(sequence, return_tensors="pt")
        output_repr = model(inputs)
        mean_repr[i] = output_repr.logits[0].mean(dim=0).detach().numpy()
    return mean_repr

def calculate_chemberta_vectors(df_met):
    df_met["successfull"] = True
    df_met["ChemBERTa"] = ""

    for ind in df_met.index:
        ID, met_type, met = df_met["ID"][ind], df_met["type"][ind], df_met["metabolite"][ind]
        if met_type == "invalid":
            mol = None
        elif met_type == "KEGG":
            try:
                mol = Chem.MolFromMolFile(join(BASE_DIR, "data", "mol-files",  met + ".mol"))
                inchi = Chem.MolToInchi(mol)
                mol = Chem.MolFromInchi(inchi)
            except:
                print(".......Mol file for KEGG ID '%s' is not available. Try to enter InChI string or SMILES for the metabolite instead." % met)
                mol = None
        elif met_type == "InChI":
            mol = Chem.inchi.MolFromInchi(met)
        elif met_type == "SMILES":
            mol = Chem.MolFromSmiles(met)
            inchi = Chem.MolToInchi(mol)
            mol = Chem.MolFromInchi(inchi)
        if mol is None:
            df_met["successfull"][ind] = False
        else:
            smiles = Chem.MolToSmiles(mol)[:510];
            df_met["ChemBERTa"][ind] = mean_repr = getMeanRepr(np.array([smiles]))
    return(df_met)


def calculate_metabolite_similarity(df_metabolites, mol):
    fp = Chem.RDKFingerprint(mol)
    
    fps = list(df_metabolites["Sim_FP"])
    IDs = list(df_metabolites["ID"])
    similarity_vector = np.zeros(len(fps))
    for i in range(len(fps)):
        similarity_vector[i] = DataStructs.FingerprintSimilarity(fp,fps[i])
    if max(similarity_vector) == 1:
        k = np.argmax(similarity_vector)
        ID = IDs[k]
    else:
        ID = np.nan
    return(max(similarity_vector), ID)