# Spotify Lyric Artist Prediction

Lauren Sun - sunlaure@usc.edu

In this project, I used the Spotify API to retrieve a list of songs by a few artists, then used [a separate API]([url](https://github.com/akashrchandran/spotify-lyrics-api)) to retrieve the lyrics for each of those songs. I then fine-tuned BERT to predict the artist given a song's lyrics.

## The Dataset

I used some code and a Flask web app from a personal project to send requests to the Spotify API. The code for data collection can be found in the "Spotify_Lyric_Data_Collection.ipynb" notebook. I first queried artists' albums, then each album's tracks, then called the lyric API on each using their unique Spotify ID. I compiled everything into a dataframe, which I exported to save. The lyrics The dataset I used to train the model can be found [here]([url](https://docs.google.com/spreadsheets/d/1NEJ3cnmpfPXntzquzYCNg2UfuDqRrvpDToN7icopaHY/edit?usp=sharing)https://docs.google.com/spreadsheets/d/1NEJ3cnmpfPXntzquzYCNg2UfuDqRrvpDToN7icopaHY/edit?usp=sharing). The lyrics are label 0 for Green Day and 1 for The Strokes.

In the data collection notebook, I originally tried to compile songs by 6 artists and thus have 6 labels for the model to learn. However, there were some issues like some artists having way larger discographies than other artists did, the Spotify API limiting each response to 50 tracks at atime, an issue with either the Spotify API or the lyric API that returned lyrics not actually by the artist, and some other difficulties, I ultimately reduced the dataset to just 2 artists. 

Another problem I wasn't sure how to approach was what exactly would consist a "sentence" to feed the model. Some songs had lengths over 1000, even though the limit for BERT is 512 tokens. I chose to just drop the few songs that had lyrics with lengths over 512 for this project. I'm considering trying this again by either splitting long song into thirds and halves. The other option would be treating each line like a sentence, which would both increase the size of the dataset and shorten the inputs to lengths closer to actual normal sentences that BERT was trained on. I decided against this because I figured that was makes artists' lyrics unique to them would be more clear when considering songs as a whole instead of single lines, but I'll have to see if that approach outperforms this attempt.

## Model Development and Training

Given that this was a binary classification task using NLP, I used the L4 notebook that fine-tuned BERT as a template to train this model. 

## Model Evaluation/Results

