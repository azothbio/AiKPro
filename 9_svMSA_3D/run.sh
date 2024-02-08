# python main2.py --gpu '0,1,2,3' --df '../Data/kpro_data_train.csv' --data '../Data/KPro_Train' --result './Result/KPro.json' --log  './log/KPro' --model_name './Model/KPro' --kfold 5

# python main.py --gpu '0,1,2,3' --df '../Data/Kinase_Extrapolation/kpro_data_train.csv' --data '../Data/Kinase_Extrapolation/KPro_Train' --result './Result/Kinase_Extrapolation/KPro.json' --log  './log/Kinase_Extrapolation/KPro' --model_name './Model/Kinase_Extrapolation/KPro' --kfold 5

python main.py --gpu '0,1,2,3' --df '../Data/SMILES_Extrapolation/kpro_data_train.csv' --data '../Data/SMILES_Extrapolation/KPro_Train' --result './Result/SMILES_Extrapolation/KPro.json' --log  './log/SMILES_Extrapolation/KPro' --model_name './Model/SMILES_Extrapolation/KPro' --kfold 5
