import requests
from textblob import TextBlob
from dash import html
import dash_bootstrap_components as dbc
import os

def get_stock_news(ticker):
    """Fetch news articles for a stock and analyze sentiment."""
    try:
        api_key = os.environ.get('NEWS_API_KEY', '0f70464767b54f46aed5f709e3961908')
        
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&pageSize=5"
        response = requests.get(url)
        

        if response.status_code != 200:
            return []

        news_data = response.json()

        news_articles = []
        if "articles" in news_data:
            for article in news_data["articles"]:
                title = article["title"]
                url = article["url"]
                description = article.get("description", "")
                published_at = article.get("publishedAt", "").split("T")[0]
                
                # Analyze sentiment
                sentiment_score = TextBlob(title).sentiment.polarity
                if sentiment_score > 0:
                    sentiment = "Positive"
                    sentiment_class = "sentiment-positive"
                    icon = html.I(className="fas fa-arrow-up me-2")
                elif sentiment_score < 0:
                    sentiment = "Negative"
                    sentiment_class = "sentiment-negative"
                    icon = html.I(className="fas fa-arrow-down me-2")
                else:
                    sentiment = "Neutral"
                    sentiment_class = "sentiment-neutral"
                    icon = html.I(className="fas fa-minus me-2")
                
                
                news_articles.append(
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span(published_at, className="text-muted float-end"),
                            html.Div([icon, sentiment], className=sentiment_class)
                        ]),
                        dbc.CardBody([
                            html.H5(title, className="news-title mb-3"),
                            html.P(description[:150] + "..." if len(description) > 150 else description, 
                                   className="text-muted mb-3"),
                            dbc.Button("Read More", href=url, target="_blank", color="primary", 
                                       className="read-more-btn")
                        ])
                    ], className="news-card shadow-sm")
                )
        
        return news_articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
