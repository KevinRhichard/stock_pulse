import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import statsmodels  
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Function to get stock news
def get_stock_news(stock_symbol):
    url = f'https://finance.yahoo.com/quote/{stock_symbol}?p={stock_symbol}'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [headline.get_text() for headline in soup.find_all('h3')]
        headlines = headlines[:10]  # Limit to the first 10 headlines
        df_news = pd.DataFrame({'Headlines': headlines})
        return df_news
    else:
        st.warning(f"Failed to retrieve data. Status code: {response.status_code}")
        return pd.DataFrame()

# Function to get historical stock data
def get_previous_year_data(stock_symbol):
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.warning(f"Failed to retrieve historical data for {stock_symbol}. Please check the stock symbol and try again.")
        return pd.DataFrame()

# Function to analyze sentiment
def analyze_sentiment(text):
    text = ' '.join(text)
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to get stock sentiment
def get_stock_sentiment(df):
    # Calculate sentiment counts
    df['Sentiment'] = df['Headlines'].apply(analyze_sentiment)
    positive_count = (df['Sentiment'] == 'Positive').sum()
    negative_count = (df['Sentiment'] == 'Negative').sum()
    neutral_count = (df['Sentiment'] == 'Neutral').sum()

    # Determine color and message based on sentiment counts
    if positive_count > negative_count:
        color = 'green'
        message = f"The reviews of the stock are looking good!"
    elif negative_count > positive_count:
        color = 'red'
        message = f"The reviews of the stock are looking bad!"
    else:
        color = 'yellow'
        message = f"The reviews of the stock are looking neutral."

    return color, message

# Function to perform demand forecasting using ARIMA
def arima_demand_forecast(stock_data, forecast_period):
    # Use closing prices as demand data
    demand_data = stock_data['Close'].values

    # Train-test split
    train, test = train_test_split(demand_data, train_size=len(demand_data) - forecast_period)

    # Fit ARIMA model
    model = ARIMA(train, order=(5, 1, 0))  # You can adjust the order based on your data
    model_fit = model.fit()

    # Forecast future demand
    forecast = model_fit.forecast(steps=forecast_period)

    return forecast

def ets_demand_forecast(stock_data, forecast_period):
    model = ExponentialSmoothing(stock_data['Close'], trend='add', seasonal='add', seasonal_periods=30)
    fit = model.fit()
    forecast = fit.forecast(steps=forecast_period)
    return forecast

# Function to plot demand data and forecast
def plot_demand_forecast(stock_data, forecast):
    fig, ax = plt.subplots()
    ax.plot(stock_data['Close'], label='Closing Prices', color='blue')
    ax.plot(pd.date_range(start=stock_data.index[-1], periods=len(forecast) + 1, freq='B')[1:], forecast, label='Forecast', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Stock Price Forecasting using ETS')
    ax.legend()
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Stock Analysis and Forecasting App")
    st.write("Enter the stock symbol:")

    # Get user input for the stock symbol
    stock_symbol = st.text_input("Stock Symbol").upper()

    if st.button("Submit"):
        # Display news headlines
        st.write("News Headlines for", stock_symbol)
        df_news = get_stock_news(stock_symbol)
        st.write(df_news)

        # Analyze sentiment
        color, message = get_stock_sentiment(df_news)
        st.markdown(f'<p style="color:{color}">{message}</p>', unsafe_allow_html=True)

        # Get historical stock data
        st.write(f"{stock_symbol} Stock Price Trend (Previous Year)")
        stock_data = get_previous_year_data(stock_symbol)
        if not stock_data.empty:
            # Plot the stock trend
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stock_data['Close'], label=f'{stock_symbol} Closing Price')
            ax.set_title(f'{stock_symbol} Stock Price Trend (Previous Year)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price (USD)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Perform stock price forecasting
            forecast_period = st.number_input('Enter the forecast period (in days):', min_value=1, max_value=365, value=90)
            forecast = ets_demand_forecast(stock_data, forecast_period)

            # Plot the demand forecast
            plot_demand_forecast(stock_data, forecast)
        else:
            st.warning(f"No historical data found for {stock_symbol}. Please check the stock symbol and try again.")

if __name__ == "__main__":
    main()
