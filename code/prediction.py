import numpy as np
import pandas as pd
import xgboost as xgb
from metabolite_preprocessing import *
from protein_representations import *
import warnings
warnings.filterwarnings("ignore")
from os.path import join
from util import validate_enzyme
import json


def SPOT_predicton(substrate_list, protein_list):
	try:
		#creating input matrices for all substrates:
		print("Step 1/3: Calculating numerical representations (ChemBERTas) for all metabolites.")

		df_met = metabolite_preprocessing_chemberta(metabolite_list = substrate_list)
		df_met = df_met.loc[df_met["successfull"]] 

		print("Step 2/3: Calculating numerical representations for all proteins.")
		protein_list = ["" if pd.isnull(value) else value for value in protein_list]

		valid_proteins, invalid_proteins = validate_enzymes(protein_list)
		df_protein = calcualte_esm1b_vectors(protein_list = valid_proteins)

		print("Step 3/3: Making predictions.")
		#Merging the Metabolite and the protein DataFrame:
		df_TS = pd.DataFrame(data = {"substrate" : substrate_list, "protein" : protein_list, "index" : list(range(len(substrate_list)))})
		df_TS = merging_metabolite_and_protein_df(df_met, df_protein, df_TS)
		df_TS_valid, df_TS_invalid = df_TS.loc[df_TS["complete"]], df_TS.loc[~df_TS["complete"]]
		df_TS_valid.reset_index(inplace = True, drop = True)

		#Making predictions
		if len(df_TS_valid) > 0:
			X = calculate_xgb_input_matrix(df = df_TS_valid)
			TSs = predict_TS(X)
			df_TS_valid["Prediction"] = list(np.round(np.array(TSs), 3))
		else:
			print("No valid input found.")

		df_TS = pd.concat([df_TS_valid, df_TS_invalid], ignore_index = True)
		df_TS = df_TS.sort_values(by = ["index"])
		df_TS.drop(columns = ["index"], inplace = True)
		df_TS.reset_index(inplace = True, drop = True)
		
		df_TS = process_df_columns(df_TS)
	
		return(df_TS)
	
	except Exception as e:
		error_message = str(e)
		print("Error:" + error_message)
		return(None)





def process_df_columns(df):
	df.drop(columns = ["ChemBERTa", "protein rep", "metabolite_similarity_score"], inplace = True)
	df.rename(columns = {"complete" : "valid input", "substrate" : "Molecules",
							"Prediction" : "Prediction score", "protein" : "Proteins"}, inplace = True)
	return(df)

def validate_enzymes(enzyme_list):
	valid_enzymes = []
	invalid_enzymes = []
	for enzyme in enzyme_list:
		if validate_enzyme(enzyme):
			valid_enzymes.append(enzyme)
		else:
			invalid_enzymes.append(enzyme)
	return(valid_enzymes, invalid_enzymes)



def merging_metabolite_and_protein_df(df_met, df_protein, df_TS):
	df_TS["ChemBERTa"], df_TS["protein rep"] = "", ""
	df_TS["complete"] = True
	df_TS["metabolite_similarity_score"] = np.nan
	df_TS["metabolite in training set"] = False
	df_TS["#metabolite in training set"] = 0
	for ind in df_TS.index:
		try:
			gnn_rep = list(df_met["ChemBERTa"].loc[df_met["metabolite"] == df_TS["substrate"][ind]])[0]
			esm1b_rep = list(df_protein["protein rep"].loc[df_protein["amino acid sequence"] == df_TS["protein"][ind]])[0]
		except:
			gnn_rep, esm1b_rep = "", ""
			df_TS["complete"][ind] = False

		if df_TS["complete"][ind]:
			df_TS["ChemBERTa"][ind] = gnn_rep
			df_TS["protein rep"][ind] = esm1b_rep
			df_TS["metabolite_similarity_score"][ind] = list(df_met["metabolite_similarity_score"].loc[df_met["metabolite"] == df_TS["substrate"][ind]])[0]
			if df_TS["metabolite_similarity_score"][ind] == 1:
				df_TS["metabolite in training set"][ind] = True
				df_TS["#metabolite in training set"][ind] = list(df_met["#metabolite in training set"].loc[df_met["metabolite"] == df_TS["substrate"][ind]])[0]
	return(df_TS)


def merging_metabolite_and_protein_df_kegg(df_met, df_protein, df_TS):
	df_TS["ChemBERTa"], df_TS["protein rep"] = "", ""
	df_TS["complete"] = True
	for ind in df_TS.index:
		try:
			gnn_rep = list(df_met["ChemBERTa"].loc[df_met["metabolite"] == df_TS["substrate"][ind]])[0]
			esm1b_rep = list(df_protein["protein rep"].loc[df_protein["amino acid sequence"] == df_TS["protein"][ind]])[0]
		except:
			gnn_rep, esm1b_rep = "", ""
			df_TS["complete"][ind] = False

		if df_TS["complete"][ind]:
			df_TS["ChemBERTa"][ind] = gnn_rep
			df_TS["protein rep"][ind] = esm1b_rep
	return(df_TS)



def predict_TS(X):
	bst= xgb.Booster()
	bst.load_model(join(BASE_DIR, "data", "xgboost_model_production_mode.dat"))
	feature_names = ['ChemBERTa_' + str(i) for i in range(767)]
	feature_names = feature_names + ['ESM1b_ts_' + str(i) for i in range(1280)]
	dX = xgb.DMatrix(X, feature_names =feature_names)
	ESs = bst.predict(dX)
	return(ESs)


def calculate_xgb_input_matrix(df):
	ESM1b = np.array(list(df["protein rep"]))
	fingerprints = ();
	for ind in df.index:
		chemberta = np.array(list(df["ChemBERTa"][ind]))[0]
		fingerprints = fingerprints +(chemberta, );
	fingerprints = np.array(fingerprints)

	print(fingerprints.shape, ESM1b.shape)
	X = np.concatenate([fingerprints, ESM1b], axis = 1)
	return(X)