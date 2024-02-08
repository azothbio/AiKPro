# python prep.py --cpu 100 --file '/home/sujeong/AiKPro/NSCLC/prediction.csv' --data './test/test' --pc

python predict.py --gpu '0,1,2,3' --df '/home/sujeong/AiKPro/NSCLC/prediction.csv' --data './test/test' --load_result './Result/KPro_1.json,./Result/KPro_2.json,./Result/KPro_3.json,./Result/KPro_4.json,./Result/KPro_5.json' --load_models './Model/KPro_1,./Model/KPro_2,./Model/KPro_3,./Model/KPro_4,./Model/KPro_5' --result  './test/Test'

