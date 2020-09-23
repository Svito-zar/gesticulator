# Gesticulator demonstration

In this folder we provide a pretrained model and a simple script for synthesizing gestures from speech and text.

Both audio and text input must be provided in order to generate new gestures.

The text input can be:
  - time-annotated JSON text transcriptions from Google Speech-to-Text (this is what the model was trained on);
  - plain .txt files containing the transcription; or
  - the transcription as a string, provided on the command-line
  - Please make sure that the text has correct punctuation, especially that it **ends with a punctuation mark**. In the case JSON files **the last "word" field of each "transcription" segment should end with a punctuation mark (? . or !).**

- In order to use the demo script below, please first install the dependencies as described in the README in the main folder of the repo.


You can find one input option in the `input` folder and you can download more example from [KTH Box](https://kth.box.com/s/hjjawinrozhg0jna2izh3kuyi95et7cg).

## Instructions

```bash
python demo.py --audio input/jeremy_howard.wav --text input/jeremy_howard.json
```

Please note that the model was trained using time-annotated JSON text transcriptions from Google Speech-to-Text, but `demo.py` accepts plaintext transcriptions as well:

```bash
python demo.py --audio input/jeremy_howard.wav --text input/jeremy_howard.txt
```

or

```bash
python demo.py --audio input/jeremy_howard.wav --text "Deep learning is an algorithm inspired by how the human brain works, and as a result it's an algorithm which has no theoretical limitations on what it can do. The more data you give it and the more computation time you give it, the better it gets. The New York Times also showed in this article another extraordinary result of deep learning which I'm going to show you now. It shows that computers can listen and understand."
```

## Output example:
[![Example video](https://i.imgur.com/OPY3zHO.png)](https://vimeo.com/449190061)
