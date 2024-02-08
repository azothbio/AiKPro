# python prep.py --cpu 100 --file '../Data/kpro_data_train.csv' --data '../Data/KPro_Train'
# python prep.py --cpu 100 --file '../Data/kpro_data_test.csv' --data '../Data/KPro_Test' --pc

python prep.py --cpu 100 --file '../Data/Kinase_Extrapolation/kpro_data_train.csv' --data '../Data/Kinase_Extrapolation/KPro_Train'
python prep.py --cpu 100 --file '../Data/Kinase_Extrapolation/kpro_data_test.csv' --data '../Data/Kinase_Extrapolation/KPro_Test' --pc

python prep.py --cpu 100 --file '../Data/SMILES_Extrapolation/kpro_data_train.csv' --data '../Data/SMILES_Extrapolation/KPro_Train'
python prep.py --cpu 100 --file '../Data/SMILES_Extrapolation/kpro_data_test.csv' --data '../Data/SMILES_Extrapolation/KPro_Test' --pc
