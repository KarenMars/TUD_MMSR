CATMuG by team 5.

## Model information
### Setup
- Add a system variable to your PC with the path `$PROJECT$\fluidsynth\bin\fluidsynth.exe`
- Import the required packages by using the `requirements.txt` file
### Run the code
- Run `catmug.py` with the desired options (use `-h` to see all options)
    - `--song $SONG_PATH$` where the song path needs to follow a certain structure such as:
    `maroon-5/payphone` which is artist/song. The music generator will then use this song to generate music using the first bar of the song
    - `--va $VALENCE$ $AROUSAL$` where the valence and arousal values are between -1.0 and 1.0, when providing these values the music generator will pick a song that is closest in the VA domain and then uses its first bar to generate music
### Data structure used
- Songs are stored in the `\data\` folder
    - the `\data\XML\` folder contains the XML files which are used as input for the music generation model
    - the `\data\midi\` folder contains the midi files for the input songs which are used for the VA predictions
    - the `\data\gen\` folder contains the midi files for the generated songs using the first bar of the valid songs as a primer
    - all of the folders in `\data\` are structured as follows: `FIRST_LETTER_ARTIST\ARTIST\SONG\`
- `\valid_songs.csv` contains all valid songs with the VA values of these songs and the VA values of their generated counterparts
- `\fluidsynth\` contains the necessary files for the FluidSynth package to be used for the VA predictions and the automatic music playback for the generated music
- `\va_predictor\` contains the VA predictor model code as well as the dataset used to train the VA regression models including the actually trained models
- `\evaluation\` contains the code for evaluating the music generator and the VA predictor and the datasets (input)


## Music generator info

This is an Implementation of MidiNet by pytorch.

MidiNet paper : https://arxiv.org/abs/1703.10847 

MidiNet code  : https://github.com/RichardYang40148/MidiNet 

dataset is from theorytab : https://www.hooktheory.com/theorytab 

You can find crawler here : https://github.com/wayne391/Symbolic-Musical-Datasets 

### What are procedures of data preprocess? 
- Firstly, the MIDI tabs which contain chords other than the **24 basic chord triads** are filtered out. 
- Secondly, for melodies (128 by 16): 
    - the smallest note unit is the sixteenth note. Thus, **the width of the melody is 16 (w=16)**. Notes which have a pause note after them are prolonged. If the first note of a bar is a pause, the second note of it is extended. Notes which are shorter than the sixteen note are exclude. 
    - Meanwhile, for simplicity, all melodies are shifted into two octaves, from C4 to B5. 
    - Although the melodies preprocessed can be represented by 24 possible notes, **all the 128 MIDI notes (h=128)** are used in our symbolic representation. The reason for that is we can detect model collapsing more easily, by checking whether the model generates notes outside these octaves. 
- Thirdly, for chords (13-by-1):
    - We use **a 13 dimensions vector** instead of using a 24-dimensional one-hot vector. The reason for that is it is more efficient. The first 12 dimensions represents the key, and that last represents the chord type (i.e. major or minor).

### Basic Model
<img src="models.png" height="350">

### Generator
#### How to incorporate conditions into generator?
In the paper **MIDINET**, it shows that it have three models of different type of conditions. 

- Model 1: melody generator, no chord condition
- Model 2: melody generator with chord condition
- Model 3: melody generator with chord condition, creative mode

The basic model is the implementation of model 3. The model takes into account of the condition of **the melody of the previous bar** and **the chord of the melody**. 
<p align="center">
<img src="CGAN.png" height="350">
</p>
The melody of the previous bar is called 2-D condition in the model/paper. The chord of the aforementioned melody is called 1-D condition in the model/paper. 

- Firstly, the 2-D condition (previous melody) is reshaped into a 1-by-2 matrix through 4 convolution layers. 
- Secondly, concatenate the white noise with 1-D condition (chord) as the input of the two linear layers. 
- Thirdly, concatenate the input of every transposed convolution layers with the 1-D condition and the transposed 2-condition. 

### Discriminator
- Only 1-D condition and the real/generated melodies are used as the input of the discriminator, the 2-D condition is not used. 


### Possible Improvements
- Try model 2, melody generator with chord condition, the 2-D previous-bar condition is only used in the last transposed convolution layer of G. The paper points out that "model 2 is more chord-dominant and stable, for it would mostly follow the chord progression and seldom generate notes that violate the constraint imposed by the chords."

## VA predictor model info
### MUSIC EMOTION RECOGNITION AND CLASSIFICATION WITH AUDIO DATA

The VA predictor is an adjusted implementation from [MUSIC MOOD DETECTION BASED ON AUDIO AND LYRICS WITH DEEP NEURAL NET](http://ismir2018.ircam.fr/doc/pdfs/99_Paper.pdf). The baseline paper is "Music  Mood  Detection  Based  On  Audio  And  Lyrics With Deep Neural Net". R. Delbouys et al tried to solve Music Emotion Recognition problem using two CNN layers and two dense layers using Deezer's music database and lyrics.

### Learning code
First, Feature extraction using mel-spectogram in src
```
$ feature_extraction.py
```   
Check hparams.py and change a parameters, and take a train_test base on your task
```
$ train_test_Regression.py
$ train_test_Classification.py
```

## Dataset used to train the VA models
### VGMIDI Dataset 

Repo: https://github.com/lucasnfe/vgmidi  
Description:
VGMIDI is a dataset of 200 MIDI labelled piano pieces (video game soundtracks). 
Annotation strategy: Each piece was annotated by 30 human subjects according to a valence-arousal model of emotion. The authors ask the annotators to write two to three sentences describing the short pieces they listened to. 
Annotation (dimensional): Time-continuous arousal and valence annotation.