import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import ta

pairs = {
    'BTC/USD': 'BTC-USD',
    'XAU/USD': ' 'GC=F',
    'GBP/JPY': 'GBPJPY=X',
    'EUR/NZD': 'EURNZD=X',
    'EUR/CAD': 'EURCAD=X'
}

app = dash.Dash(__name__)
app.title = "Analyse Forex/Crypto"

app.layout = html.Div([
    html.H2("Analyse Automatique des Paires Forex/Crypto"),
    dcc.Dropdown(
        id='pair-selector',
        options=[{'label': k, 'value': v} for k, v in pairs.items()],
        value='BTC-USD',
        style={'width': '300px'}
    ),
    html.Button('Analyser', id='analyze-button', n_clicks=0, style={'marginTop': '10px'}),
    html.Div(id='results', style={'marginTop': '20px'}),
    dcc.Graph(id='chart'),
    html.A("Voir le calendrier économique", href="https://www.investing.com/economic-calendar/", target="_blank", style={"marginTop": "20px", "display": "block"})
])

@app.callback(
    Output('results', 'children'),
    Output('chart', 'figure'),
    Input('analyze-button', 'n_clicks'),
    Input('pair-selector', 'value')
)
def update_analysis(n_clicks, symbol):
    if not symbol:
        return "Sélectionnez une paire", go.Figure()

    df = yf.download(symbol, period="6mo", interval="1d")
    if df.empty:
        return "Données non disponibles.", go.Figure()
    df.dropna(inplace=True)
    close = df['Close'].squeeze()

    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['SMA_50'] = ta.trend.SMAIndicator(close, 50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(close, 200).sma_indicator()
    bb = ta.volatility.BollingerBands(close, 20, 2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()

    def detect_levels(df, window=5):
        levels = []
        lows = df['Low'].values
        highs = df['High'].values
        dates = df.index
        for i in range(window, len(df) - window):
            low = lows[i]
            high = highs[i]
            if all(low < lows[i - j] for j in range(1, window + 1)) and all(low < lows[i + j] for j in range(1, window + 1)):
                levels.append((dates[i], float(low)))
            if all(high > highs[i - j] for j in range(1, window + 1)) and all(high > highs[i + j] for j in range(1, window + 1)):
                levels.append((dates[i], float(high)))
        return levels

    levels = detect_levels(df)
    entry = float(df['Close'].iloc[-1])
    supports = [lvl[1] for lvl in levels if lvl[1] < entry]
    resistances = [lvl[1] for lvl in levels if lvl[1] > entry]
    sl = min(supports) if supports else entry * 0.95
    tp = max(resistances) if resistances else entry * 1.05
    rr = round(abs(tp - entry) / abs(entry - sl), 2) if sl != entry else None
    progress = round((entry - sl) / (tp - sl) * 100, 2) if tp != sl else 0

    alerts = []
    if df['RSI'].iloc[-1] > 70:
        alerts.append("RSI en surachat (>70)")
    elif df['RSI'].iloc[-1] < 30:
        alerts.append("RSI en survente (<30)")
    if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
        alerts.append("MACD haussier (MACD > Signal)")
    elif df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]:
        alerts.append("MACD baissier (MACD < Signal)")

    infos = html.Div([
        html.P(f"Entrée : {entry:.2f} | SL : {sl:.2f} | TP : {tp:.2f}"),
        html.P(f"Ratio R/R : {rr} | Progression : {progress}%"),
        html.Ul([html.Li(alert) for alert in alerts]) if alerts else html.P("Aucune alerte.")
    ])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Prix'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='Bollinger Haut'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='Bollinger Bas'))

    for date, level in levels:
        fig.add_shape(type='line', x0=date, x1=date,
                      y0=level * 0.98, y1=level * 1.02,
                      line=dict(color="purple", width=1, dash="dot"))

    fig.update_layout(title=f"Analyse - {symbol}", xaxis_title="Date", yaxis_title="Prix")
    return infos, fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)