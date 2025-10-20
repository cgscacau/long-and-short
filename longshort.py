import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from datetime import datetime, timedelta
import time
import requests
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Long & Short Strategy Analyzer",
    page_icon="📈",
    layout="wide"
)

# Título e descrição
st.title("📈 Long & Short Strategy Analyzer")
st.markdown("""
Analise estratégias de Long & Short com ações brasileiras. 
Selecione os ativos, período e visualize correlações, retornos e performance da estratégia.
""")

# Funções auxiliares para download de dados

def download_from_investing(ticker, start_date, end_date):
    """
    Tenta baixar dados do Investing.com via scraping
    """
    try:
        # Mapeamento de tickers B3 para Investing.com
        ticker_map = {
            'PETR4.SA': 'PETR4',
            'VALE3.SA': 'VALE3',
            'ITUB4.SA': 'ITUB4',
            'BBDC4.SA': 'BBDC4',
            'ABEV3.SA': 'ABEV3',
            'ELET3.SA': 'ELET3',
            'BBAS3.SA': 'BBAS3',
            'WEGE3.SA': 'WEGE3',
            'RADL3.SA': 'RADL3',
            'LREN3.SA': 'LREN3',
            'MGLU3.SA': 'MGLU3',
            'SUZB3.SA': 'SUZB3',
            'JBSS3.SA': 'JBSS3',
            'B3SA3.SA': 'B3SA3',
            'RENT3.SA': 'RENT3'
        }
        
        clean_ticker = ticker_map.get(ticker, ticker.replace('.SA', ''))
        
        # URL do Yahoo Finance com formato CSV
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {
            'period1': int(start_date.timestamp()),
            'period2': int(end_date.timestamp()),
            'interval': '1d',
            'events': 'history',
            'includeAdjustedClose': 'true'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
            if not df.empty and 'Close' in df.columns:
                return df['Close']
        
    except Exception as e:
        pass
    
    return None

def download_from_brapi(ticker, start_date, end_date):
    """
    Baixa dados da API brasileira brapi.dev (gratuita)
    """
    try:
        clean_ticker = ticker.replace('.SA', '')
        
        # Formatar datas
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # URL da API brapi
        url = f"https://brapi.dev/api/quote/{clean_ticker}"
        params = {
            'range': '1y',  # Pode ajustar conforme necessário
            'interval': '1d',
            'fundamental': 'false'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                
                if 'historicalDataPrice' in result:
                    hist_data = result['historicalDataPrice']
                    
                    # Converter para DataFrame
                    df = pd.DataFrame(hist_data)
                    df['date'] = pd.to_datetime(df['date'], unit='s')
                    df.set_index('date', inplace=True)
                    df = df.sort_index()
                    
                    # Filtrar pelo período
                    df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
                    
                    if not df.empty and 'close' in df.columns:
                        return df['close']
        
    except Exception as e:
        pass
    
    return None

def download_synthetic_data(ticker, start_date, end_date):
    """
    Gera dados sintéticos realistas para demonstração
    """
    try:
        # Número de dias
        days = (end_date - start_date).days
        
        # Gerar datas
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Preço inicial baseado no ticker
        ticker_prices = {
            'PETR4.SA': 35.0,
            'VALE3.SA': 65.0,
            'ITUB4.SA': 25.0,
            'BBDC4.SA': 15.0,
            'ABEV3.SA': 12.0,
            'ELET3.SA': 40.0,
            'BBAS3.SA': 30.0,
            'WEGE3.SA': 45.0,
            'RADL3.SA': 25.0,
            'LREN3.SA': 18.0,
            'MGLU3.SA': 3.0,
            'SUZB3.SA': 50.0,
            'JBSS3.SA': 30.0,
            'B3SA3.SA': 12.0,
            'RENT3.SA': 60.0
        }
        
        initial_price = ticker_prices.get(ticker, 50.0)
        
        # Parâmetros de volatilidade por setor
        volatility_map = {
            'PETR4.SA': 0.025,  # Petróleo - alta volatilidade
            'VALE3.SA': 0.022,  # Mineração - alta volatilidade
            'ITUB4.SA': 0.015,  # Banco - média volatilidade
            'BBDC4.SA': 0.015,  # Banco - média volatilidade
            'ABEV3.SA': 0.012,  # Bebidas - baixa volatilidade
            'ELET3.SA': 0.020,  # Energia - média volatilidade
            'BBAS3.SA': 0.015,  # Banco - média volatilidade
            'WEGE3.SA': 0.018,  # Industrial - média volatilidade
            'RADL3.SA': 0.014,  # Varejo farmacêutico - baixa volatilidade
            'LREN3.SA': 0.020,  # Varejo - média/alta volatilidade
            'MGLU3.SA': 0.035,  # E-commerce - muito alta volatilidade
            'SUZB3.SA': 0.022,  # Papel e celulose - alta volatilidade
            'JBSS3.SA': 0.020,  # Alimentos - média volatilidade
            'B3SA3.SA': 0.018,  # Bolsa - média volatilidade
            'RENT3.SA': 0.016   # Locação - média volatilidade
        }
        
        volatility = volatility_map.get(ticker, 0.02)
        
        # Gerar retornos com tendência leve
        np.random.seed(hash(ticker) % 2**32)  # Seed baseado no ticker para consistência
        
        # Adicionar tendência sutil (drift)
        drift = 0.0003  # 0.03% ao dia em média
        
        returns = np.random.normal(drift, volatility, len(dates))
        
        # Adicionar autocorrelação (efeito momentum)
        for i in range(1, len(returns)):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Calcular preços
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Criar Series
        price_series = pd.Series(prices, index=dates, name=ticker)
        
        return price_series
        
    except Exception as e:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data_multi_source(tickers, start_date, end_date, use_synthetic=False):
    """
    Tenta baixar dados de múltiplas fontes com fallback
    """
    all_data = {}
    failed_tickers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"📥 Processando {ticker}... ({idx + 1}/{total_tickers})")
        progress_bar.progress((idx + 1) / total_tickers)
        
        data = None
        
        if not use_synthetic:
            # Método 1: Yahoo Finance com requests direto
            status_text.text(f"📥 {ticker} - Tentando Yahoo Finance... ({idx + 1}/{total_tickers})")
            data = download_from_investing(ticker, start_date, end_date)
            
            if data is not None and len(data) > 0:
                all_data[ticker] = data
                time.sleep(0.5)
                continue
            
            time.sleep(1)
            
            # Método 2: API brapi.dev
            status_text.text(f"📥 {ticker} - Tentando API brasileira... ({idx + 1}/{total_tickers})")
            data = download_from_brapi(ticker, start_date, end_date)
            
            if data is not None and len(data) > 0:
                all_data[ticker] = data
                time.sleep(0.5)
                continue
            
            time.sleep(1)
        
        # Método 3: Dados sintéticos (sempre disponível)
        status_text.text(f"📥 {ticker} - Gerando dados sintéticos... ({idx + 1}/{total_tickers})")
        data = download_synthetic_data(ticker, start_date, end_date)
        
        if data is not None and len(data) > 0:
            all_data[ticker] = data
            if not use_synthetic:
                st.info(f"ℹ️ Usando dados sintéticos para {ticker} (dados reais indisponíveis)")
        else:
            failed_tickers.append(ticker)
            st.warning(f"⚠️ Não foi possível obter dados para {ticker}")
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        return None, failed_tickers
    
    # Combinar dados em um DataFrame
    df = pd.DataFrame(all_data)
    
    # Remover linhas com muitos NaN
    df = df.dropna(thresh=len(df.columns) * 0.5)
    
    # Forward fill para preencher gaps pequenos
    df = df.ffill(limit=5)
    
    return df, failed_tickers

def calculate_returns(prices):
    """Calcula retornos percentuais diários."""
    return prices.pct_change().dropna()

def calculate_statistics(returns):
    """Calcula estatísticas descritivas dos retornos."""
    if len(returns) == 0:
        return None
    
    stats = {
        'Retorno Médio Diário (%)': returns.mean() * 100,
        'Retorno Anualizado (%)': returns.mean() * 252 * 100,
        'Volatilidade Diária (%)': returns.std() * 100,
        'Volatilidade Anualizada (%)': returns.std() * np.sqrt(252) * 100,
        'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
        'Retorno Acumulado (%)': ((1 + returns).cumprod().iloc[-1] - 1) * 100 if len(returns) > 0 else 0,
        'Máxima Drawdown (%)': calculate_max_drawdown(returns) * 100
    }
    return stats

def calculate_max_drawdown(returns):
    """Calcula o máximo drawdown."""
    if len(returns) == 0:
        return 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_correlation_matrix(returns):
    """Calcula matriz de correlação."""
    return returns.corr()

def run_regression(y, x):
    """Executa regressão linear entre dois ativos."""
    valid_idx = ~(x.isna() | y.isna())
    x_clean = x[valid_idx]
    y_clean = y[valid_idx]
    
    if len(x_clean) < 10:
        return None
    
    X = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, X).fit()
    return model

def calculate_hedge_ratio(long_returns, short_returns):
    """Calcula o hedge ratio ótimo usando regressão."""
    model = run_regression(long_returns, short_returns)
    if model is None:
        return 1.0
    return model.params[1]

def calculate_long_short_strategy(long_prices, short_prices, hedge_ratio=1.0):
    """Calcula retornos da estratégia Long & Short."""
    long_returns = calculate_returns(long_prices)
    short_returns = calculate_returns(short_prices)
    
    common_idx = long_returns.index.intersection(short_returns.index)
    long_returns = long_returns.loc[common_idx]
    short_returns = short_returns.loc[common_idx]
    
    strategy_returns = long_returns - (hedge_ratio * short_returns)
    
    return strategy_returns

# Sidebar - Configurações
st.sidebar.header("⚙️ Configurações")

# Modo de dados
st.sidebar.subheader("🔧 Fonte de Dados")
data_mode = st.sidebar.radio(
    "Escolha a fonte:",
    ["Tentar dados reais (pode falhar)", "Usar dados sintéticos (sempre funciona)"],
    help="Dados sintéticos são gerados algoritmicamente para demonstração"
)

use_synthetic = data_mode == "Usar dados sintéticos (sempre funciona)"

if use_synthetic:
    st.sidebar.info("ℹ️ Modo demonstração ativado. Os dados são gerados algoritmicamente com características realistas de mercado.")

# Lista de ações brasileiras populares
default_stocks = [
    'PETR4.SA',   # Petrobras PN
    'VALE3.SA',   # Vale ON
    'ITUB4.SA',   # Itaú PN
    'BBDC4.SA',   # Bradesco PN
    'ABEV3.SA',   # Ambev ON
    'ELET3.SA',   # Eletrobras ON
    'BBAS3.SA',   # Banco do Brasil ON
    'WEGE3.SA',   # WEG ON
    'RADL3.SA',   # Raia Drogasil ON
    'LREN3.SA'    # Lojas Renner ON
]

# Seleção de ativos
st.sidebar.subheader("📊 Seleção de Ativos")

use_custom = st.sidebar.checkbox("Usar lista customizada", value=False)

if use_custom:
    custom_input = st.sidebar.text_area(
        "Digite os tickers (um por linha):",
        value="PETR4.SA\nVALE3.SA\nITUB4.SA",
        height=150,
        help="Digite cada ticker em uma nova linha"
    )
    selected_stocks = [t.strip().upper() for t in custom_input.split('\n') if t.strip()]
else:
    selected_stocks = st.sidebar.multiselect(
        "Escolha os ativos:",
        options=default_stocks,
        default=['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'],
        help="Selecione pelo menos 2 ativos"
    )

# Período de análise
st.sidebar.subheader("📅 Período de Análise")

period_preset = st.sidebar.selectbox(
    "Período rápido:",
    ["Customizado", "1 Mês", "3 Meses", "6 Meses", "1 Ano", "2 Anos"],
    index=3
)

if period_preset == "Customizado":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Data Inicial:",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "Data Final:",
            value=datetime.now(),
            max_value=datetime.now()
        )
else:
    end_date = datetime.now().date()
    period_map = {
        "1 Mês": 30,
        "3 Meses": 90,
        "6 Meses": 180,
        "1 Ano": 365,
        "2 Anos": 730
    }
    start_date = end_date - timedelta(days=period_map[period_preset])

if start_date >= end_date:
    st.sidebar.error("❌ Data inicial deve ser anterior à data final!")
    st.stop()

# Configurações da estratégia Long & Short
st.sidebar.subheader("🎯 Estratégia Long & Short")

if len(selected_stocks) >= 2:
    long_asset = st.sidebar.selectbox(
        "Ativo LONG:",
        options=selected_stocks,
        index=0
    )
    
    short_options = [s for s in selected_stocks if s != long_asset]
    short_asset = st.sidebar.selectbox(
        "Ativo SHORT:",
        options=short_options,
        index=0 if short_options else None
    )
    
    use_optimal_hedge = st.sidebar.checkbox(
        "Usar hedge ratio ótimo",
        value=True,
        help="Calcula o hedge ratio usando regressão linear"
    )
    
    if not use_optimal_hedge:
        hedge_ratio = st.sidebar.slider(
            "Hedge Ratio manual:",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1
        )

# Botão para executar análise
run_analysis = st.sidebar.button("🚀 Executar Análise", type="primary", use_container_width=True)

# Botão para limpar cache
if st.sidebar.button("🔄 Limpar Cache", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache limpo!")

# Main content
if run_analysis:
    if len(selected_stocks) < 2:
        st.error("❌ Selecione pelo menos 2 ativos para análise!")
        st.stop()
    
    # Download dos dados
    st.info("📥 Iniciando processamento dos dados...")
    
    prices_df, failed = download_stock_data_multi_source(
        selected_stocks,
        start_date,
        end_date,
        use_synthetic=use_synthetic
    )
    
    if prices_df is None or prices_df.empty:
        st.error("❌ Não foi possível obter dados para os ativos selecionados.")
        st.stop()
    
    if failed and not use_synthetic:
        st.warning(f"⚠️ Alguns tickers falharam: {', '.join(failed)}")
    
    available_stocks = list(prices_df.columns)
    
    if len(available_stocks) < 2:
        st.error("❌ Menos de 2 ativos disponíveis.")
        st.stop()
    
    st.success(f"✅ Dados obtidos com sucesso para {len(available_stocks)} ativos!")
    
    # Verificar ativos da estratégia
    if long_asset not in available_stocks:
        long_asset = available_stocks[0]
        st.info(f"ℹ️ Ativo LONG ajustado para: {long_asset}")
    
    if short_asset not in available_stocks:
        short_candidates = [s for s in available_stocks if s != long_asset]
        if short_candidates:
            short_asset = short_candidates[0]
            st.info(f"ℹ️ Ativo SHORT ajustado para: {short_asset}")
        else:
            short_asset = None
    
    returns_df = calculate_returns(prices_df)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral",
        "📈 Análise de Preços",
        "🔗 Correlações",
        "💼 Estratégia Long & Short"
    ])
    
    # TAB 1: Visão Geral
    with tab1:
        st.header("Estatísticas Descritivas")
        
        stats_data = {}
        for ticker in prices_df.columns:
            stats = calculate_statistics(returns_df[ticker])
            if stats:
                stats_data[ticker] = stats
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data).T
            
            st.dataframe(
                stats_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=0),
                use_container_width=True
            )
            
            st.subheader("📊 Métricas Principais")
            
            cols = st.columns(min(len(prices_df.columns), 4))
            for i, ticker in enumerate(prices_df.columns):
                col_idx = i % 4
                with cols[col_idx]:
                    if ticker in stats_data:
                        st.metric(
                            label=ticker,
                            value=f"{stats_data[ticker]['Retorno Acumulado (%)']:.2f}%",
                            delta=f"{stats_data[ticker]['Retorno Anualizado (%)']:.2f}% anual"
                        )
        
        st.info(f"""
        **Período analisado:** {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')}  
        **Dias úteis:** {len(returns_df)}  
        **Ativos analisados:** {len(prices_df.columns)}
        """)
    
    # TAB 2: Análise de Preços
    with tab2:
        st.header("Evolução dos Preços")
        
        normalized_prices = (prices_df / prices_df.iloc[0]) * 100
        
        fig = go.Figure()
        
        for ticker in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker],
                mode='lines',
                name=ticker,
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Preços Normalizados (Base 100)",
            xaxis_title="Data",
            yaxis_title="Preço Normalizado",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Retornos Acumulados")
        
        cumulative_returns = (1 + returns_df).cumprod() - 1
        
        fig2 = go.Figure()
        
        for ticker in cumulative_returns.columns:
            fig2.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[ticker] * 100,
                mode='lines',
                name=ticker,
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
        
        fig2.update_layout(
            title="Retornos Acumulados (%)",
            xaxis_title="Data",
            yaxis_title="Retorno Acumulado (%)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 3: Correlações
    with tab3:
        st.header("Análise de Correlações")
        
        corr_matrix = calculate_correlation_matrix(returns_df)
        
        fig4 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlação")
        ))
        
        fig4.update_layout(
            title="Matriz de Correlação dos Retornos",
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        st.dataframe(
            corr_matrix.style.format("{:.3f}").background_gradient(cmap='RdBu', vmin=-1, vmax=1),
            use_container_width=True
        )
    
    # TAB 4: Estratégia Long & Short
    with tab4:
        st.header("Análise da Estratégia Long & Short")
        
        if short_asset is None:
            st.error("❌ Não há ativos suficientes para estratégia")
        elif long_asset not in prices_df.columns or short_asset not in prices_df.columns:
            st.error("❌ Ativos selecionados não disponíveis")
        else:
            if use_optimal_hedge:
                long_ret = returns_df[long_asset]
                short_ret = returns_df[short_asset]
                optimal_hedge = calculate_hedge_ratio(long_ret, short_ret)
                hedge_ratio = optimal_hedge
                st.info(f"📊 Hedge Ratio Ótimo: **{hedge_ratio:.4f}**")
            else:
                st.info(f"📊 Hedge Ratio Manual: **{hedge_ratio:.4f}**")
            
            strategy_returns = calculate_long_short_strategy(
                prices_df[long_asset],
                prices_df[short_asset],
                hedge_ratio
            )
            
            if len(strategy_returns) > 0:
                strategy_stats = calculate_statistics(strategy_returns)
                
                if strategy_stats:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Retorno Acumulado", f"{strategy_stats['Retorno Acumulado (%)']:.2f}%")
                    with col2:
                        st.metric("Retorno Anualizado", f"{strategy_stats['Retorno Anualizado (%)']:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{strategy_stats['Sharpe Ratio']:.2f}")
                    with col4:
                        st.metric("Máx. Drawdown", f"{strategy_stats['Máxima Drawdown (%)']:.2f}%")
                    
                    st.subheader("Performance da Estratégia")
                    
                    strategy_cumulative = (1 + strategy_returns).cumprod() - 1
                    long_cumulative = (1 + returns_df[long_asset]).cumprod() - 1
                    short_cumulative = (1 + returns_df[short_asset]).cumprod() - 1
                    
                    fig6 = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Retornos Acumulados', 'Drawdown'),
                        vertical_spacing=0.15,
                        row_heights=[0.7, 0.3]
                    )
                    
                    fig6.add_trace(
                        go.Scatter(
                            x=strategy_cumulative.index,
                            y=strategy_cumulative * 100,
                            mode='lines',
                            name='Estratégia L&S',
                            line=dict(color='purple', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    fig6.add_trace(
                        go.Scatter(
                            x=long_cumulative.index,
                            y=long_cumulative * 100,
                            mode='lines',
                            name=f'Long: {long_asset}',
                            line=dict(dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    fig6.add_trace(
                        go.Scatter(
                            x=short_cumulative.index,
                            y=short_cumulative * 100,
                            mode='lines',
                            name=f'Short: {short_asset}',
                            line=dict(dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    cumulative_strategy = (1 + strategy_returns).cumprod()
                    running_max = cumulative_strategy.expanding().max()
                    drawdown = (cumulative_strategy - running_max) / running_max * 100
                    
                    fig6.add_trace(
                        go.Scatter(
                            x=drawdown.index,
                            y=drawdown,
                            mode='lines',
                            name='Drawdown',
                            fill='tozeroy',
                            line=dict(color='red')
                        ),
                        row=2, col=1
                    )
                    
                    fig6.update_xaxes(title_text="Data", row=2, col=1)
                    fig6.update_yaxes(title_text="Retorno (%)", row=1, col=1)
                    fig6.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
                    
                    fig6.update_layout(
                        height=700,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig6, use_container_width=True)
                    
                    st.subheader("Comparação de Performance")
                    
                    long_stats = calculate_statistics(returns_df[long_asset])
                    short_stats = calculate_statistics(returns_df[short_asset])
                    
                    if long_stats and short_stats:
                        comparison_data = {
                            'Estratégia L&S': strategy_stats,
                            f'Long: {long_asset}': long_stats,
                            f'Short: {short_asset}': short_stats
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data).T
                        
                        st.dataframe(
                            comparison_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=0),
                            use_container_width=True
                        )

else:
    st.info("👈 Configure os parâmetros e clique em **'Executar Análise'**")
    
    st.markdown("""
    ### 📚 Como usar:
    
    **Fonte de Dados:**
    - **Tentar dados reais**: Tenta baixar do Yahoo Finance e APIs brasileiras
    - **Usar dados sintéticos**: Gera dados realistas para demonstração (sempre funciona)
    
    **Seleção de Ativos:**
    - Escolha pelo menos 2 ativos da lista
    - Ou use a opção customizada
    
    **Análises Disponíveis:**
    - 📊 Estatísticas e métricas de performance
    - 📈 Evolução de preços e retornos
    - 🔗 Matriz de correlação
    - 💼 Performance da estratégia Long & Short
    
    ### 💡 Dica:
    
    Se você está tendo problemas com dados reais, use o **modo sintético** para testar a ferramenta.
    Os dados sintéticos são gerados com características realistas de mercado!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📈 Long & Short Strategy Analyzer</p>
    <p style='font-size: 0.8em; color: gray;'>Desenvolvido com Streamlit | Apenas para fins educacionais</p>
</div>
""", unsafe_allow_html=True)
