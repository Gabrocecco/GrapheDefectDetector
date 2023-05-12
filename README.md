# GrapheDefectDetector

1) Trasformazione dataset .xyz in .png ✔️
2) Object Detection sui difetti ✔️
3) Analisi dei difetti con OpenCV per estrapolazione feature geometriche ✔️
4) Correlazione feature geometriche dei difetti con propietà fisiche del campione ✔️
5) Modello predittivo per total_energy a partire dalle feature geomtriche dei campioni ✔️

E' possibile lanciare una demo passo passo da main_playground.ipynb

Traformazione campioni in formato .xyz in immagini:

![slide](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/5a37633a-c502-4b3e-8c18-bc3ab3daf749)

![b74c9ce2-graphene_218641_bonds](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/b5efe18b-ed51-467b-8599-eab58c0a24bf)


![graphene_4734_bonds_cropped_box_2_thresh_](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/f75aaa39-8712-4a2f-a787-7d5da8061b81)

![OjectDetect_example](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/bee7ab69-bdf7-4688-8120-2c4c3dc77370)

Il training di YOLO è stato fatto con 100 immagini su colab. 



![val_batch0_pred](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/ca3e7231-ee7b-48ea-b325-2c92717a0d58)

Il test del preditorre è stato fatto con 1000 inferenze di YOLO. 



![fit](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/e4dc38f7-03e4-4fbf-89ba-1b8cd2765340)


![var](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/d4b6c8ec-715f-49b8-ad5f-e516c71d88b6)

![mdi](https://github.com/Gabrocecco/GrapheDefectDetector/assets/52239001/2cd9b203-e7e8-4467-a22e-8bd2a9d12ec5)
