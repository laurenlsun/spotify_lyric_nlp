# Spotify Lyric Artist Prediction

Lauren Sun - sunlaure@usc.edu

In this project, I used the Spotify API to retrieve a list of songs by a few artists, then used [a separate API](https://github.com/akashrchandran/spotify-lyrics-api) to retrieve the lyrics for each of those songs. I preprocessed and tokenized the words in the BERT-specific format. I then fine-tuned BERT to predict the artist given a song's lyrics.

## The Dataset

I used some code and a Flask web app from a personal project to send requests to the Spotify API; the code for data collection can be found in the "Spotify_Lyric_Data_Collection.ipynb" notebook. I first queried artists' albums, then each album's tracks, then called the lyric API on each using their unique Spotify ID. I compiled everything into a dataframe, which I exported to save. The dataset I used to train the model can be found [here](https://docs.google.com/spreadsheets/d/1NEJ3cnmpfPXntzquzYCNg2UfuDqRrvpDToN7icopaHY/edit?usp=sharing). The lyrics are labeled 0 for Green Day and 1 for The Strokes.

In the data collection notebook, I originally tried to compile songs by 6 artists and thus have 6 labels for the model to learn. However, there were some issues like some artists having way larger discographies than other artists did, the Spotify API limiting each response to 50 tracks at atime, an issue with either the Spotify API or the lyric API that returned lyrics not actually by the artist, issues with cuda, etc, so I ultimately reduced the dataset to just 2 artists. 

Another problem I wasn't sure how to approach was what exactly would consist a "sentence" to feed the model. Some songs had lengths over 1000, even though the limit for BERT is 512 tokens. I chose to just drop the few songs that had lyrics with lengths over 512. 

## Model Development and Training

Given that this was a binary classification task using NLP, I used the L4 notebook that fine-tuned BERT as a template to train this model. The small size of my dataset would be less of an issue because BERT has already been pre-trained. Training the extra layer on top of BERT would produce good results with relatively minimal effort. 

When playing around with parameters, I noticed the model did better as I increased learning rate, which I initially saw as a good sign. But considering how small the dataset is, I think the model started overfitting the small dataset rather than actually picking up on the lyrics' features that could indicate which artist wrote them. 

## Model Evaluation/Results

I tested the model with lyrics it had never seen. I used the Matthews correlation coefficient (MCC) because the test set was slightly imbalanced (though the test set itself was only a sample of 25). 

batch_size=16, MCC = 0.592, 80% accuracy
![image](https://github.com/laurenlsun/spotify_lyric_nlp/assets/119720461/5cf75b62-b19d-45b3-8c11-57b5ae0fdefd)

batch_size=32, MCC = 0.612, 76% accuracy
![image](https://github.com/laurenlsun/spotify_lyric_nlp/assets/119720461/0cb57ffa-4fa1-4fb1-82d5-8ca8ea3d7b9b)

The model, with the best combination of hyperparameters I could find, was able to label new lyrics with the correct artist with accuracy % in the high 70s-low 80s.

## Discussion

Interestingly, though their MCCs were close, training with a batch size of 32 made the model overwhelmingly predict Green Day songs to be Strokes songs (and be slightly less accurate), whereas a batch size of 16 did not. This is an example of the MCC not capturing all the nuances of the model's performance.

I think the way I chose to treat entire songs as a single sentence is rather problematic, and I'd definitely try other ways of breaking up songs into pieces. I'm considering either splitting long song into thirds and halves or treating each line like a sentence, which would both increase the size of the dataset and shorten the inputs to lengths closer to actual normal sentences that BERT was trained on. I originally decided against this because I figured that whatever makes artists' lyrics unique to them would be more clear when considering songs as a whole instead of single lines. I also want to fix the API and cuda issues I ran into when trying to implement multiple classes as opposed to just two and see if making this a multi-class labeling task changes anything. Given additional computing power, I'd also like to try training for longer periods of time, though I have doubts as to whether it would improve its performance. I don't think it's an easy task even for humans.

Overfitting is a pretty big issue in this specific lyric classification. After all, is there really a whole lot to look for on a semantic level when predicting which artist wrote a set of lyrics? The model might start memorizing the dataset rather than find some underlying tell-tale features hidden in the lyrics because maybe there just aren't such features. It'd be easy for the model to look too closely at the lyrics for features to predict the artist when, all things considered, it probably isn't that deep. 
