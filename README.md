## GrapheDefectDetector

# R^2 score del predittore: 0.97

1) Trasformazione dataset .xyz in .png ✔️
2) Object Detection sui difetti ✔️
3) Analisi dei difetti con OpenCV per estrapolazione feature geometriche ✔️
4) Correlazione feature geometriche dei difetti con propietà fisiche del campione ✔️
5) Modello predittivo per total_energy a partire dalle feature geomtriche dei campioni ✔️

E' possibile lanciare una demo passo passo da main_playground.ipynb

Traformazione campioni in formato .xyz in immagini:

![thumbnail_campione_to_png](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/6d7e50d8-a0fd-4582-966b-676cf2fb2fb6)
![val_batch0_pred](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/46ec382d-0ae8-4ba7-8b2b-5c4a8896c97c)
Trasformazioni sull'immagine per evidenziare l'area:
![graphene_67_bonds_cropped_box_1_thresh_](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/1f56d6b6-c464-492f-a675-d62ddbed182e)
Estrazione fetaures grafiche e correllazione: 
![graphene_67_bonds_cropped_box_1_thresh_countour_](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/5ac5eb80-67f6-4d8d-807a-b6320095acc4)

![heatmap](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/a97d72d4-7697-419d-bd2d-45f25417e4a4)
![corr](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/3ce7598c-d442-4c2d-8dfa-b777cec7d076)


Il training di YOLO è stato fatto con 100 immagini su colab. 
Il test del preditorre è stato fatto con 1000 inferenze di YOLO. 

![var](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/45a96ff0-0c03-44bc-9d60-6f21f7bf9247)
![fit](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/66d7e8e5-3eb3-4687-adea-a0640888c9f7)
![mdi](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/9acb884c-5e75-472d-93de-0c22d05c50e3)
