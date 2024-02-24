# End-to-end machine Translation from Scratch
This project implements a neural machine translation web application from scratch using transformers and Streamlit.

### Overview
The app translates text between English and Italian using a transformer-based sequence-to-sequence model. The model is trained on the Opus books English-Italian dataset.

A Streamlit interface allows users to enter text and translate between languages. The translated text is displayed along with an attention visualization to show word alignments.

### Model
The neural machine translation model is implemented in PyTorch using the Transformer architecture. It contains an encoder and decoder with multi-head self-attention layers.

The model is trained with label-smoothed cross entropy loss and achieves a BLEU score of 34 on the test set.

### Web Interface
The web interface is implemented using Streamlit. The user can enter text in the source language. The model translates to the target language and displays the output.

The interface is responsive and the model achieves low latency on translation allowing for a smooth user experience.


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/nikhil0035/Machine_Translation_using_Transformers
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n NMT python=3.10 -y
```

```bash
conda activate NMT
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
streamlit run app.py
```

Now,
```bash
open up your local host and port
```

### Future Work
Potential areas for improvement:
Support more languages
Improve model accuracy with back translation
Add user accounts to customize translations
Deploy web app on cloud infrastructure for scalability

### References
Vaswani et al. Attention Is All You Need. NeurIPS 2017.
