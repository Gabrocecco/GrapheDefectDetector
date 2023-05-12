# GrapheDefectDetector

1) Trasformazione dataset .xyz in .png ✔️
2) Object Detection sui difetti ✔️
3) Analisi dei difetti con OpenCV per estrapolazione feature geometriche ✔️
4) Correlazione feature geometriche dei difetti con propietà fisiche del campione ✔️
5) Modello predittivo per total_energy a partire dalle feature geomtriche dei campioni ✔️

E' possibile lanciare una demo passo passo da main_playground.ipynb

Traformazione campioni in formato .xyz in immagini:

![slide](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/5a37633a-c502-4b3e-8c18-bc3ab3daf749)
![OjectDetect_example](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/bee7ab69-bdf7-4688-8120-2c4c3dc77370)

Trasformazioni sull'immagine per evidenziare l'area:
![graphene_4734_bonds_cropped_box_2_thresh_](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/f75aaa39-8712-4a2f-a787-7d5da8061b81)

Estrazione fetaures grafiche e correllazione: 

![heatmap](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/c875ab62-cf52-4179-818f-93b7d1284e65)
![corr](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/a6862c38-39da-4079-9d1d-e97ebb612b25)

Il training di YOLO è stato fatto con 100 immagini su colab. 
Il test del preditorre è stato fatto con 1000 inferenze di YOLO. 

![fit](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/e4dc38f7-03e4-4fbf-89ba-1b8cd2765340)
![var](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/d4b6c8ec-715f-49b8-ad5f-e516c71d88b6)
![mdi](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/2cd9b203-e7e8-4467-a22e-8bd2a9d12ec5)
