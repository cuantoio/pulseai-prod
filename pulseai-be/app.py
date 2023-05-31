# support api
from flask import Flask, jsonify, request
from datetime import datetime, timedelta, date
from joblib import Parallel, delayed
from dotenv import load_dotenv
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import openai
import boto3
import time
import json
import re 
import os

# product api
import pulseGPT

app = Flask(__name__)
CORS(app)

load_dotenv()

openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

MEMORY_SIZE = 10
memory = []
stock_data_cache = {} 

@app.route('/')
def run_test():
    print('live')
    return 'live'

s3 = boto3.client('s3')
BUCKET_NAME = os.getenv("BUCKET_NAME")
CHAT_HISTORY_PREFIX = 'chat_history/'

def save_chat_history(chat_history):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    chat_history_key = f"{CHAT_HISTORY_PREFIX}chat_history_{today}.json"
    
    print('save chat_history::',chat_history)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=chat_history_key,
        Body=json.dumps(chat_history)
    )

def load_chat_history():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    chat_history_key = f"{CHAT_HISTORY_PREFIX}chat_history_{today}.json"
    
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=chat_history_key)
        chat_history = json.loads(response['Body'].read())
        print("Loaded chat history:", chat_history)  # Add this line to print the loaded chat history
        
        # Filter out non-dictionary entries
        chat_history = [chat for chat in chat_history if isinstance(chat, dict)]

        return chat_history[:7]
    except s3.exceptions.NoSuchKey:
        return []

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.get_json()
    print("Received data:", data)

    message_id = data.get('message_id')
    feedback = data.get('feedback')
    content = data.get('content')
    keywords = data.get('keywords')
    summary = data.get('summary')
    sentiment = data.get('sentiment')
    gpt_summary = data.get('gpt_summary')
    prompt = data.get('prompt')
    response = data.get('response')
    time = data.get('time')
    username = data.get('username')

    if message_id and feedback:
        feedback_data = {
            'message_id': message_id,
            'feedback': feedback,
            'content': content,
            'keywords': keywords,
            'summary': summary,
            'sentiment': sentiment,
            'gpt_summary': gpt_summary,
            'prompt': prompt,
            'response': response,
            'time': time,
            'username': username
        }

        feedback_key = f"feedback/{message_id}.json"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=feedback_key,
            Body=json.dumps(feedback_data)
        )
        print('success', feedback_data)
        return jsonify({'status': 'success'})
    else:
        return jsonify({'error': 'Invalid input'}), 400

def simulate_single_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe_ratio, weights

def get_key_allocations(results, weights_record, stock_data):
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights_record[max_sharpe_idx], index=stock_data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[1])
    sdp_min, rp_min = results[1, min_vol_idx], results[0, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights_record[min_vol_idx], index=stock_data.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    return max_sharpe_allocation, min_vol_allocation

def simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    with Parallel(n_jobs=-1, prefer="threads") as parallel:
        results_and_weights = parallel(
            delayed(simulate_single_portfolio)(mean_returns, cov_matrix, risk_free_rate)
            for _ in range(num_portfolios)
        )

    results = np.array([item[0:3] for item in results_and_weights]).T
    weights_record = [item[3] for item in results_and_weights]

    return results, weights_record

PROFILE_PREFIX = 'user_profile/'

def save_user_portfolio(portfolio, username):
    portfolio_key = f"{PROFILE_PREFIX}{username}_profile.json"
    
    s3.put_object(
    Bucket=BUCKET_NAME,
    Key=portfolio_key,
    Body=json.dumps(portfolio)
    )

def load_user_profile(username, default={}):
    profile_key = f"{PROFILE_PREFIX}{username}_profile.json"

    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=profile_key)
        user_profile = json.loads(response['Body'].read())
        print("Loaded user profile:", user_profile) 
        
        return user_profile
    except s3.exceptions.NoSuchKey:
        return default

@app.route('/api/efficient_frontier', methods=['POST'])
def efficient_frontier():
    data = request.json
    username = data.get('username', 'tsm')
    prompt = data.get('prompt', '').lower()

    user_profile = load_user_profile(username, {})

    # Parse tickers from the prompt
    prompt_tickers = re.findall(r'\$(?:\s*|,\s*)(\w+)', prompt)
    prompt_tickers = [ticker.upper() for ticker in prompt_tickers]

    # Add new tickers to the portfolio with default allocation
    for ticker in prompt_tickers:
        if ticker not in loaded_user_portfolio:
            loaded_user_portfolio[ticker] = {"allocation": 0}  # default allocation

    print('ef_prompt::',prompt)
    print('ef_user_profile::',user_profile)
    print('ef_prompt_tickers::',prompt_tickers)
    print('ef_loaded_user_portfolio::',loaded_user_portfolio)

    if 'new' in prompt:
        # Reset portfolio
        loaded_user_portfolio = {}
        
        # Get tickers after 'new' keyword
        new_tickers = re.findall(r'new\s+(.*)', prompt)[0].split()
        new_tickers = [ticker.lstrip('$').upper() for ticker in new_tickers]
        for ticker in new_tickers:
            loaded_user_portfolio[ticker] = {"allocation": 0}  # default allocation
    
    elif loaded_user_portfolio or 'my' in prompt or 'load' in prompt:

        print('ef_if::', "in 'my' or 'load' elif")
        # Load existing portfolio
        loaded_user_portfolio = user_profile
    else:

        print('ef_if::', "in else")
        # Load default portfolio
        loaded_user_portfolio = {                    
            "AAPL": {"allocation": 35.86},
            "META": {"allocation": 8.52}, 
            "MSFT": {"allocation": 28.08},
            "TSLA": {"allocation": 25.71},
        }

    if 'add' in prompt:        

        print('ef_if::', "in 'add' if")
        loaded_user_portfolio = {
            "NVDA": {"allocation": 22},
            "META": {"allocation": 13},
            "MSFT": {"allocation": 7},
            "GOOG": {"allocation": 5}, 
            "SPY": {"allocation": 5}, 
        }
                
    # Parse the prompt for 'remove' command and tickers
    elif 'remove' in prompt:

        print('ef_if::', "in 'remove' if")
        removed_tickers = prompt.split('remove ')[1].split()
        for ticker in removed_tickers:
            if ticker.upper() in loaded_user_portfolio:
                del loaded_user_portfolio[ticker.upper()]

    tickers = list(loaded_user_portfolio.keys())
    allocations = [v["allocation"] for v in loaded_user_portfolio.values()]

    if not tickers:
        return jsonify({'message': 'No tickers found'}), 400

    stock_data_key = tuple(sorted(tickers))
    if stock_data_key not in stock_data_cache:
        stock_data_cache[stock_data_key] = yf.download(tickers, start='2022-05-17', end='2023-05-17')['Adj Close']

    stock_data = stock_data_cache[stock_data_key]
    daily_returns = stock_data.pct_change()
    daily_returns = daily_returns.dropna()

    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    risk_free_rate = 0.045
    num_portfolios = 10000
    results, weights_record = simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_allocation, min_vol_allocation = get_key_allocations(results, weights_record, stock_data)

    max_sharpe_index = results[2, :].argmax()
    min_vol_index = results[1, :].argmin()

    print("ef_max_sharpe_allocation:: ", max_sharpe_allocation.to_dict())
    print("ef_min_volatility_allocation:: ", min_vol_allocation.to_dict())

    # Parse the prompt for 'save' command 
    if 'save' in prompt:
        save_user_portfolio(max_sharpe_allocation.to_dict(), username)
        return jsonify({'message': 'Portfolio saved successfully'}), 200

    return jsonify({
        'tickers': tickers,
        'allocations': allocations,
        'max_sharpe_allocation': max_sharpe_allocation.to_dict(),
        'min_vol_allocation': min_vol_allocation.to_dict(),
        'results': results.tolist(),
        'weights_record': [w.tolist() for w in weights_record],
        'max_sharpe_index': int(max_sharpe_index),
        'min_vol_index': int(min_vol_index),
    })
            
@app.route('/api/combined_summary', methods=['POST'])
def api_combined_summary():
    data = request.get_json()
    query = data.get('query')
    num_results = data.get('num_results', 10)
    username = data.get('username', 'tsm')
    user_profile = load_user_profile(username)

    user_prompt = {
        'timestamp': time.time(),
        'sender': username,
        'message': query
    }
    
    print('cs_user_prompt::',user_prompt)
    print('cs_user_profile::',user_profile)

    if query != "":
        gpt_prompt = f"Remember to respond in a tone similar to the input prompt in 250 characters or less. You can provide more details on request: {user_profile}'{query}'\n\n"
        
        portfolio = f"""AAPL: 36.53%
TSLA: 28.60%
MSFT: 26.19%
META: 8.68%"""

        gpt_prompt += f"\n portfolio {portfolio}\n"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI portfolio manager. You are equipped with hedge fund grade skills to balance and optimize investment portfolios for users. You automatically update the chart based on the optimized values. respond in this format: response $ TICKER1 TICKER2",
                },
                {"role": "user", "content": gpt_prompt},
            ],
        )

        gpt_response = response.choices[0].message['content'].strip()
        return jsonify({'combined_summary': gpt_response, 'user_profile': user_profile})

    else:  # Query is empty, use Google Search instead.
        combined_summary = "Can you provide more details on what you're looking to do?" #get_combined_summary(query, num_results)
        return jsonify({'combined_summary': combined_summary, 'user_profile': user_profile})
    
if __name__ == "__main__":
    # app.run(port=5000)
    app.run(host="0.0.0.0", port=8080)
