# Final-Project---Public Sentiment Solar Power Incentives in California 
Description: This document is the Final Project of my Python Class - Sentiment Analysis of California's Policy on Solar Panels

Energy transition has always been a big topic for today. With the increasing impacts and influences of climate change, more people in the world know about the importance of reducing carbon emissions and being more sustainable. Industries and transportation sectors are both gradually transitioning their energy sources from fossil fuels to electricity. For individuals, many families, especially in California and other states with long sunshine duration, adopted household solar panels to generate their own electricity at home. They consider many factors like cost, policy incentives, long long-term benefits, in whether to buy solar panels for their own house. Policies in the US like net metering which is a pricing mechanism and many other incentives attract some people to make the decision. However, the effectiveness of the policy and the attitude of Californian residents in installing solar panels need to be examined carefully. The final project aims to do sentiment analysis on California's incentives for solar panels to find out the public opinion towards government action in energy policy. The analysis of the public sentiment will contribute to the policy implications across the nation. 

The research question of the project is how public sentiment reacts to a decline in California's solar power incentives. The hypothesis is that a decline in California's solar power incentives can lead to a shift in public sentiment from positive to negative because fewer incentives towards the adoption of solar panels can raise problems of financial barriers to installation, loss of confidence in government, and diminished trust in renewable energy initiatives in the long term. Therefore, the project is essential in evaluating the sentiments and providing suggestions regarding policy implications. It can affect California's long-term clean energy transition in the long term. Since California is the most active state in solar energy, its policies also have the potential to impact domestic solar energy policy in decades. 

The method of the project is a text analysis of the comments under YouTube videos talking about the decline in California's incentives for solar panels. The Video talks about the shift in the government's actions on solar panels, the opinions of people who adopted solar panels before, the attitude of people who choose not to buy panels, and the explanations of policies from government officials. The video contains almost all opinions from all stakeholders so people who watched this video are well-informed about the situation. Also, more than 300 comments are under this video, making it a good starting point for data collection for the project. 

The method for data analysis for public sentiments includes pandas, API key, nltk, nltk.sentiment.vader, SentimentIntensityAnalyzer, matplotlib.pyplot, and wordcloud. YouTube Data API key is essential for Python to get information accessed to video. After the access is confirmed, Pandas are used for organizing, manipulating, and cleaning datasets of 300 comments in tabular format to help future analysis. The nltk is the natural language toolkit which is crucial for text processing tasks. The function helps to remove stopwords, make tokenization, and prepare data for sentiment analysis. Vader and Intensity Analyzer calculate the sentiment score for each comment. A negative score means negative sentiment. A positive score means positive sentiments. 0 represents neutral sentiments. The matplotlib.pyplot is used for data visualizations like bar charts and distribution. The outcome is more direct by using this command. Finally, wordcloud is a way to visualize the most frequent word. From wordcloud, it is easier to see which word is most cared for or concerned by people who have a strong sentiment. 

The project starts with accessing the data from Youtube by using YouTube Data API key.
```python
from googleapiclient.discovery import build
import pandas as pd

api_key = 'AIzaSyC2omyULB21L1GTH84Ti32CA6x4o5Ilt6o'
youtube = build('youtube', 'v3', developerKey=api_key)
```

This concise function retrieves all comments from a specified YouTube video by iterating through all pages using the YouTube Data API. 

```python
def get_comments(video_id):
    comments = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        ).execute()
        comments += [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments

video_id = '5vQmk2oQnfQ'
comments = get_comments(video_id)
print(f"Total comments fetched: {len(comments)}")
```
Total comments fetched: 309

Then an API key is used to fetch comments from a specific YouTube video and prints the first 10 comments.

```python
api_key = 'AIzaSyC2omyULB21L1GTH84Ti32CA6x4o5Ilt6o'

video_id = '5vQmk2oQnfQ'
comments = get_comments(video_id)

for comment in comments[:10]:
    print(comment)
```
The first 10 comments are printed after the code.

A data frame from the comments list is created and the code provides a statistical summary of the `Comment` column. The format is good for sentiment analysis.
```python
import pandas as pd
df_comments = pd.DataFrame(comments, columns=['Comment'])
df_comments
```
Then the function is used to clean text by removing URLs, special characters, and extra spaces, converts text to lowercase, and applies it to the `Comment` column to create a new `Cleaned_Comment` column in the DataFrame.
```python
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    text = text.lower().strip()  
    return text

df_comments['Cleaned_Comment'] = df_comments['Comment'].apply(clean_text)
print(df_comments.head())
```
After cleaned_comment is done, nltk should be installed and ready for the sentiment analysis. 

```python
pip install nltk
```
Then, VADER sentiment analysis is used to assign sentiment scores and labels (Positive, Negative, Neutral) to each cleaned comment in the DataFrame. Each comment have a sentiment score after this code. 
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
df_comments['Sentiment_Score'] = df_comments['Cleaned_Comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df_comments['Sentiment_Label'] = df_comments['Sentiment_Score'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
)
print(df_comments.head())
```
Data visualization is crucial in doing sentiment analysis so matplotlib.pyplot is used to make a bar chart to show the distribution of sentiment according to the score. 

```python
import matplotlib.pyplot as plt
sentiment_counts = df_comments['Sentiment_Label'].value_counts()
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```
 The code produces the image as follows: 
 
![My Image](https://private-user-images.githubusercontent.com/181412027/397597976-020ed84d-83a9-406b-8b4b-e8247e6a3338.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzQ2NjM5MDEsIm5iZiI6MTczNDY2MzYwMSwicGF0aCI6Ii8xODE0MTIwMjcvMzk3NTk3OTc2LTAyMGVkODRkLTgzYTktNDA2Yi04YjRiLWU4MjQ3ZTZhMzMzOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMjIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTIyMFQwMzAwMDFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kN2I1NjU1ZWUzYmViZGU1N2MyYjkyODEzMGMwYWVhZjE0YjM0OTM2N2I0N2ExOGJiNWYwYmQzNjY1YWVkOTA2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.aZZl3KnhujEr5JbLrMrna20ZiTjqVH_aWHQCqbp3zJA)

The green column represents the total count of comments that have a negative sentiment (negative score).  The red column in the middle represents the total count of comments that have a positive sentiment (positive score). The blue column on the right side represents the total count of comments that have a neutral sentiment. In the distribution chart, it is obvious that the number of positive and negative comments is almost equal, around 120 counts. The neutral comments are only about 60. The equal number of positive and negative comments indicates the opinion towards the decline in incentives is highly polarized. The people with neutral comments are not too much so it is not a concern.

Based on the analysis of distribution, the analysis of keywords for both positive and negative is crucial for understanding the reason for people's sentiments. The wordcloud is used for this purpose. 

This is the code for wordcloud for positive comments.
```python
pip install wordcloud

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

custom_stopwords = {"solar", "power", "energy", "year"}

stopwords = set(STOPWORDS).union(custom_stopwords)

positive_comments = " ".join(df_comments[df_comments['Sentiment_Label'] == 'Positive']['Cleaned_Comment'])
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(positive_comments)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Comments Word Cloud')
plt.show()
```

The first wordcloud is for positive comments, as shown. 

![My Image](https://private-user-images.githubusercontent.com/181412027/397604532-7d587de6-9d55-4527-a187-fdc96e77797c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzQ2NjUxNzMsIm5iZiI6MTczNDY2NDg3MywicGF0aCI6Ii8xODE0MTIwMjcvMzk3NjA0NTMyLTdkNTg3ZGU2LTlkNTUtNDUyNy1hMTg3LWZkYzk2ZTc3Nzk3Yy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMjIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTIyMFQwMzIxMTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yYWVlYmMxMTYyOTgwNmM0NTk4ODJkMjY4Y2FmMjg4NTM2NTk2NTZmODZlMDBiNDQyMmVkNzI2ZjgwY2ZkYTc5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hrWHRBaFXqdTMQDxnSYQ4Wjr0Tx9WL78nYVMoc1M7tk)

In the wordcloud for positive comments, top words include "will", "cost", "California", and "grid". The meaning of these words represents factors of why people have positive attitudes. "Will" shows people's expectation of the long-term benefits gained from installing the solar panel and their confidence in the policy incentives in the future. "Cost" indicates that people care about the cost of the solar panel and the financial factor is one of the most important things to consider when it comes to installing solar panels. People comment "California" because they still have positive sentiment toward the state's action in making policy decisions on solar. Many "gird" in the positive comments indicate the strong relationship between the traditional grid and the solar panel. Overall, the positive comments express people's high expectations about future policy incentives made by Californian governments. Additionally, words like "solar", "power", "energy", and "year" have been defined as stopwords since they do not contain meaning in explaining the idea behind sentiments. 

This is the code for negative comments. 

```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

custom_stopwords = {"solar", "power", "energy", "electricity", "panel"}

stopwords = set(STOPWORDS).union(custom_stopwords)

negative_comments = " ".join(df_comments[df_comments['Sentiment_Label'] == 'Negative']['Cleaned_Comment'])
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(negative_comments)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Comments Word Cloud')
plt.show()
```
The second wordcloud is for negative comments, as shown. 
![My Image](https://private-user-images.githubusercontent.com/181412027/397608840-d61d76fa-43b5-4758-8997-bd9234d51eed.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzQ2NjY4MzMsIm5iZiI6MTczNDY2NjUzMywicGF0aCI6Ii8xODE0MTIwMjcvMzk3NjA4ODQwLWQ2MWQ3NmZhLTQzYjUtNDc1OC04OTk3LWJkOTIzNGQ1MWVlZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMjIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTIyMFQwMzQ4NTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mOGZkODBmOTljNzBiMzk3NTIxMDY0ZDE1YmM1OWUwNzZlNWRjNGMwYmM5Njk0ODIyNGI5MzBjYTRhMTc2NmU3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.iPRJjyuVIDHraFZ3lJuZMJpJ5jjCsRwbAPf_GkvWnGI)

In the wordcloud for negative comments, top words include "grid", "pay", "people", and "government". The word "grid" shows people's dissatisfaction with the traditional grid. The word "pay" shows that people have financial considerations about adopting solar panels, which is the same as people with positive attitudes. The word "people" indicates that commenters care about how other people feel about the policy incentives. The word "government" frequently exists in negative comments showing commenters' low confidence in government actions because of the shift in policy incentives. The words "solar", "power", "energy", "electricity", and "panel" are defined as stopwords in this wordcloud as well because they do not express further meanings. 

Overall, the public sentiment toward the government's action in shifting policy incentives for solar energy is highly polarized among people who commented on the YouTube video. While many people see opportunities and remain optimistic about solar energy despite subsidy cuts, an equal number express concerns about affordability, policy decisions, or the implications for renewable energy adoption. The issue sparks strong, divided reactions, highlighting its complexity and the need for further discussion and clarity.

Policy implications based on the project include supporting solar energy adoption, balancing budgetary and environmental goals, and monitoring adoption rates. In the purpose of achieving the clean energy transition goal, policymakers should consider introducing alternative support mechanisms to promote solar energy adoption, like tax credits, low-interest loans for solar energy use, or subsidies for low-income households to encourage solar adoption. Balancing budgetary and environmental goals is also important. The recent decline in incentives in California is partly due to budgetary issues. While cutting subsidies may be fiscally motivated, it is essential to balance these savings with California’s broader environmental goals. Ensuring a steady growth in solar adoption aligns with the state’s commitments to renewable energy and carbon reduction targets. Finally, monitoring adoption rates is a good way to react to neutral commenters so policymakers can adjust policies when a significant decline is observed. 


