# MedDiag upload test samples

These are small public research/training examples for exercising the upload flow. They are not clinical records and must not be used to make medical decisions. Prediction results from this prototype are not diagnostic.

## ECG

- `ecg_mitbih_100.csv` and `ecg_mitbih_100.txt` contain the first 10 seconds (3,600 samples) of lead MLII from record 100 of the PhysioNet MIT-BIH Arrhythmia Database. The CSV declares `sampling_rate=360`; the TXT upload should use 360 Hz.
- Source: https://physionet.org/content/mitdb/1.0.0/ — please cite the MIT-BIH Arrhythmia Database when reusing the data.
- The `.txt` and `.csv` files are equivalent upload examples prepared from the official WFDB record.

## EEG

- `Bonn_Set_Z_Z001.txt` is one 4,097-sample EEG recording from Bonn University dataset Set Z (healthy volunteer, eyes open). `Bonn_Set_Z_Z001.csv` is the same data in CSV form. Sampling rate: 173.61 Hz.
- Source/mirror and dataset details: https://github.com/NeuralPsy/Bonn-EEG-Dataset — original source: University of Bonn. For academic/research use; cite Andrzejak et al. (2001), *Physical Review E* 64, 061907, DOI: https://doi.org/10.1103/PhysRevE.64.061907.
- The dataset is single-channel and the Set Z data are scalp EEG.

## X-ray

- `chest_xray_sample.jpg` is a chest radiograph from the COVID-19 Image Data Collection, image `000001-6.jpg` (Eurorad case 1189). Its metadata lists CC BY-NC-SA 4.0. Source collection: https://github.com/ieee8023/covid-chestxray-dataset; source case: https://www.eurorad.org/case/1189.
- `chest_xray_sample.png` is a PNG conversion of that JPEG.
- `chest_xray_sample.dcm` is a DICOM Secondary Capture wrapper generated from the same image pixels for upload-format testing. It is not the source examination's original DICOM. Its DICOM patient fields are synthetic test identifiers.
- For the JPEG/PNG source, retain the attribution and share-alike terms above.
