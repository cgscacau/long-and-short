import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from datetime import datetime, timedelta
import time

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

# Funções auxiliares
@st.cache_data(ttl=3600)
def download_stock_data(tickers, start_date, end_date, max_retries=3):
    """
    Baixa dados de ações com retry logic e tratamento de erros robusto.
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
        max_retries: Número máximo de tentativas
    
    Returns:
        DataFrame com os dados ou None em caso de erro
    """
    all_data = {}
    failed_tickers = []
    
    for ticker in tickers:
        success = False
        
        for attempt in range(max_retries):
            try:
                # Download individual para cada ticker
                stock = yf.Ticker(ticker)
                data = stock.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True
                )
                
                if not data.empty and len(data) > 0:
                    all_data[ticker] = data['Close']
                    success = True
                    break
                else:
                    time.sleep(1)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    st.warning(f"⚠️ Erro ao baixar {ticker}: {str(e)}")
        
        if not success:
            failed_tickers.append(ticker)
    
    if not all_data:
        return None, failed_tickers
    
    # Combinar dados em um DataFrame
    df = pd.DataFrame(all_data)
    
    return df, failed_tickers

def calculate_returns(prices):
    """Calcula retornos percentuais diários."""
    return prices.pct_change().dropna()

def calculate_statistics(returns):
    """
    Calcula estatísticas descritivas dos retornos.
    
    Args:
        returns: Series com retornos
    
    Returns:
        Dict com estatísticas
    """
    stats = {
        'Retorno Médio Diário (%)': returns.mean() * 100,
        'Retorno Anualizado (%)': returns.mean() * 252 * 100,
        'Volatilidade Diária (%)': returns.std() * 100,
        'Volatilidade Anualizada (%)': returns.std() * np.sqrt(252) * 100,
        'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
        'Retorno Acumulado (%)': ((1 + returns).cumprod().iloc[-1] - 1) * 100,
        'Máxima Drawdown (%)': calculate_max_drawdown(returns) * 100
    }
    return stats

def calculate_max_drawdown(returns):
    """Calcula o máximo drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_correlation_matrix(returns):
    """Calcula matriz de correlação."""
    return returns.corr()

def run_regression(y, x):
    """
    Executa regressão linear entre dois ativos.
    
    Args:
        y: Retornos do ativo dependente
        x: Retornos do ativo independente
    
    Returns:
        Resultados da regressão
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model

def calculate_hedge_ratio(long_returns, short_returns):
    """Calcula o hedge ratio ótimo usando regressão."""
    model = run_regression(long_returns, short_returns)
    return model.params[1]  # Beta

def calculate_long_short_strategy(long_prices, short_prices, hedge_ratio=1.0):
    """
    Calcula retornos da estratégia Long & Short.
    
    Args:
        long_prices: Preços do ativo long
        short_prices: Preços do ativo short
        hedge_ratio: Ratio de hedge (default 1:1)
    
    Returns:
        Series com retornos da estratégia
    """
    long_returns = calculate_returns(long_prices)
    short_returns = calculate_returns(short_prices)
    
    # Estratégia: Long no primeiro ativo, Short no segundo
    strategy_returns = long_returns - (hedge_ratio * short_returns)
    
    return strategy_returns

# Sidebar - Configurações
st.sidebar.header("⚙️ Configurações")

# Lista de ações brasileiras populares
default_stocks = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'BBAS3.SA', 'WEGE3.SA', 'RENT3.SA', 'MGLU3.SA', 'B3SA3.SA'
]

# Seleção de ativos
st.sidebar.subheader("📊 Seleção de Ativos")
selected_stocks = st.sidebar.multiselect(
    "Escolha os ativos:",
    options=default_stocks,
    default=['PETR4.SA', 'VALE3.SA'],
    help="Selecione pelo menos 2 ativos para análise"
)

# Adicionar tickers customizados
custom_ticker = st.sidebar.text_input(
    "Adicionar ticker customizado:",
    placeholder="Ex: AAPL, GOOGL",
    help="Digite o ticker e pressione Enter"
)

if custom_ticker and custom_ticker not in selected_stocks:
    selected_stocks.append(custom_ticker.upper())

# Período de análise
st.sidebar.subheader("📅 Período de Análise")

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

# Validação de datas
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
            step=0.1,
            help="Proporção de hedge entre long e short"
        )

# Botão para executar análise
run_analysis = st.sidebar.button("🚀 Executar Análise", type="primary", use_container_width=True)

# Main content
if run_analysis:
    if len(selected_stocks) < 2:
        st.error("❌ Selecione pelo menos 2 ativos para análise!")
        st.stop()
    
    # Download dos dados
    with st.spinner("📥 Baixando dados do Yahoo Finance..."):
        prices_df, failed = download_stock_data(
            selected_stocks,
            start_date,
            end_date
        )
    
    if prices_df is None or prices_df.empty:
        st.error("❌ Não foi possível baixar dados para os ativos selecionados. Tente novamente mais tarde.")
        if failed:
            st.warning(f"Tickers que falharam: {', '.join(failed)}")
        st.stop()
    
    if failed:
        st.warning(f"⚠️ Alguns tickers falharam: {', '.join(failed)}")
    
    # Remover colunas com todos NaN
    prices_df = prices_df.dropna(axis=1, how='all')
    
    if prices_df.empty:
        st.error("❌ Nenhum dado válido foi obtido.")
        st.stop()
    
    # Calcular retornos
    returns_df = calculate_returns(prices_df)
    
    # Tabs para organizar conteúdo
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral",
        "📈 Análise de Preços",
        "🔗 Correlações",
        "💼 Estratégia Long & Short"
    ])
    
    # TAB 1: Visão Geral
    with tab1:
        st.header("Estatísticas Descritivas")
        
        # Calcular estatísticas para cada ativo
        stats_data = {}
        for ticker in prices_df.columns:
            stats_data[ticker] = calculate_statistics(returns_df[ticker])
        
        stats_df = pd.DataFrame(stats_data).T
        
        # Formatar DataFrame
        st.dataframe(
            stats_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=0),
            use_container_width=True
        )
        
        # Métricas principais
        st.subheader("📊 Métricas Principais")
        
        cols = st.columns(len(prices_df.columns))
        for i, ticker in enumerate(prices_df.columns):
            with cols[i]:
                st.metric(
                    label=ticker,
                    value=f"{stats_data[ticker]['Retorno Acumulado (%)']:.2f}%",
                    delta=f"{stats_data[ticker]['Retorno Anualizado (%)']:.2f}% anual"
                )
        
        # Informações do período
        st.info(f"""
        **Período analisado:** {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')}  
        **Dias úteis:** {len(returns_df)}  
        **Ativos analisados:** {len(prices_df.columns)}
        """)
    
    # TAB 2: Análise de Preços
    with tab2:
        st.header("Evolução dos Preços")
        
        # Normalizar preços para base 100
        normalized_prices = (prices_df / prices_df.iloc[0]) * 100
        
        # Gráfico de preços normalizados
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
        
        # Gráfico de retornos acumulados
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
        
        # Distribuição de retornos
        st.subheader("Distribuição de Retornos Diários")
        
        fig3 = go.Figure()
        
        for ticker in returns_df.columns:
            fig3.add_trace(go.Histogram(
                x=returns_df[ticker] * 100,
                name=ticker,
                opacity=0.7,
                nbinsx=50
            ))
        
        fig3.update_layout(
            title="Histograma de Retornos Diários",
            xaxis_title="Retorno (%)",
            yaxis_title="Frequência",
            barmode='overlay',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # TAB 3: Correlações
    with tab3:
        st.header("Análise de Correlações")
        
        # Matriz de correlação
        corr_matrix = calculate_correlation_matrix(returns_df)
        
        # Heatmap de correlação
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
        
        # Tabela de correlação
        st.subheader("Tabela de Correlações")
        st.dataframe(
            corr_matrix.style.format("{:.3f}").background_gradient(cmap='RdBu', vmin=-1, vmax=1),
            use_container_width=True
        )
        
        # Scatter plots entre pares
        if len(returns_df.columns) >= 2:
            st.subheader("Scatter Plots - Retornos Diários")
            
            col1, col2 = st.columns(2)
            
            with col1:
                asset1 = st.selectbox("Ativo 1:", returns_df.columns, key='scatter1')
            with col2:
                asset2_options = [c for c in returns_df.columns if c != asset1]
                asset2 = st.selectbox("Ativo 2:", asset2_options, key='scatter2')
            
            if asset1 and asset2:
                fig5 = go.Figure()
                
                fig5.add_trace(go.Scatter(
                    x=returns_df[asset1] * 100,
                    y=returns_df[asset2] * 100,
                    mode='markers',
                    marker=dict(size=5, opacity=0.6),
                    name='Retornos'
                ))
                
                # Linha de regressão
                model = run_regression(returns_df[asset2], returns_df[asset1])
                x_line = np.linspace(returns_df[asset1].min(), returns_df[asset1].max(), 100)
                y_line = model.params[0] + model.params[1] * x_line
                
                fig5.add_trace(go.Scatter(
                    x=x_line * 100,
                    y=y_line * 100,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name=f'Regressão (β={model.params[1]:.3f})'
                ))
                
                fig5.update_layout(
                    title=f"Relação entre {asset1} e {asset2}",
                    xaxis_title=f"Retorno {asset1} (%)",
                    yaxis_title=f"Retorno {asset2} (%)",
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig5, use_container_width=True)
                
                # Estatísticas da regressão
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Beta (β)", f"{model.params[1]:.4f}")
                with col2:
                    st.metric("R²", f"{model.rsquared:.4f}")
                with col3:
                    st.metric("Correlação", f"{returns_df[[asset1, asset2]].corr().iloc[0, 1]:.4f}")
    
    # TAB 4: Estratégia Long & Short
    with tab4:
        st.header("Análise da Estratégia Long & Short")
        
        if long_asset not in prices_df.columns or short_asset not in prices_df.columns:
            st.error("❌ Ativos selecionados não estão disponíveis nos dados baixados.")
        else:
            # Calcular hedge ratio
            if use_optimal_hedge:
                long_ret = returns_df[long_asset]
                short_ret = returns_df[short_asset]
                optimal_hedge = calculate_hedge_ratio(long_ret, short_ret)
                hedge_ratio = optimal_hedge
                st.info(f"📊 Hedge Ratio Ótimo Calculado: **{hedge_ratio:.4f}**")
            else:
                st.info(f"📊 Hedge Ratio Manual: **{hedge_ratio:.4f}**")
            
            # Calcular retornos da estratégia
            strategy_returns = calculate_long_short_strategy(
                prices_df[long_asset],
                prices_df[short_asset],
                hedge_ratio
            )
            
            # Estatísticas da estratégia
            strategy_stats = calculate_statistics(strategy_returns)
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Retorno Acumulado",
                    f"{strategy_stats['Retorno Acumulado (%)']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Retorno Anualizado",
                    f"{strategy_stats['Retorno Anualizado (%)']:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{strategy_stats['Sharpe Ratio']:.2f}"
                )
            
            with col4:
                st.metric(
                    "Máx. Drawdown",
                    f"{strategy_stats['Máxima Drawdown (%)']:.2f}%"
                )
            
            # Gráfico de performance
            st.subheader("Performance da Estratégia")
            
            # Calcular retornos acumulados
            strategy_cumulative = (1 + strategy_returns).cumprod() - 1
            long_cumulative = (1 + returns_df[long_asset]).cumprod() - 1
            short_cumulative = (1 + returns_df[short_asset]).cumprod() - 1
            
            fig6 = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Retornos Acumulados', 'Drawdown'),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            # Retornos acumulados
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
            
            # Drawdown
            strategy_dd = pd.Series(
                [calculate_max_drawdown(strategy_returns[:i+1]) * 100 
                 for i in range(len(strategy_returns))],
                index=strategy_returns.index
            )
            
            fig6.add_trace(
                go.Scatter(
                    x=strategy_dd.index,
                    y=strategy_dd,
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
            
            # Tabela comparativa
            st.subheader("Comparação de Performance")
            
            comparison_data = {
                'Estratégia L&S': strategy_stats,
                f'Long: {long_asset}': calculate_statistics(returns_df[long_asset]),
                f'Short: {short_asset}': calculate_statistics(returns_df[short_asset])
            }
            
            comparison_df = pd.DataFrame(comparison_data).T
            
            st.dataframe(
                comparison_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=0),
                use_container_width=True
            )
            
            # Análise de risco-retorno
            st.subheader("Análise Risco-Retorno")
            
            fig7 = go.Figure()
            
            for name, stats in comparison_data.items():
                fig7.add_trace(go.Scatter(
                    x=[stats['Volatilidade Anualizada (%)']],
                    y=[stats['Retorno Anualizado (%)']],
                    mode='markers+text',
                    name=name,
                    text=[name],
                    textposition='top center',
                    marker=dict(size=15)
                ))
            
            fig7.update_layout(
                title="Risco vs Retorno (Anualizado)",
                xaxis_title="Volatilidade (%)",
                yaxis_title="Retorno (%)",
                height=500,
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig7, use_container_width=True)
            
            # Rolling statistics
            st.subheader("Estatísticas Móveis (30 dias)")
            
            rolling_window = 30
            rolling_return = strategy_returns.rolling(window=rolling_window).mean() * 252 * 100
            rolling_vol = strategy_returns.rolling(window=rolling_window).std() * np.sqrt(252) * 100
            rolling_sharpe = rolling_return / rolling_vol
            
            fig8 = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Retorno Anualizado Móvel', 'Volatilidade Anualizada Móvel', 'Sharpe Ratio Móvel'),
                vertical_spacing=0.1
            )
            
            fig8.add_trace(
                go.Scatter(x=rolling_return.index, y=rolling_return, name='Retorno'),
                row=1, col=1
            )
            
            fig8.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Volatilidade'),
                row=2, col=1
            )
            
            fig8.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='Sharpe'),
                row=3, col=1
            )
            
            fig8.update_xaxes(title_text="Data", row=3, col=1)
            fig8.update_yaxes(title_text="Retorno (%)", row=1, col=1)
            fig8.update_yaxes(title_text="Volatilidade (%)", row=2, col=1)
            fig8.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
            
            fig8.update_layout(
                height=900,
                showlegend=False,
                template='plotly_white'
            )
            
            st.plotly_chart(fig8, use_container_width=True)

else:
    # Tela inicial
    st.info("👈 Configure os parâmetros na barra lateral e clique em **'Executar Análise'** para começar.")
    
    st.markdown("""
    ### 📚 Como usar esta ferramenta:
    
    **Seleção de Ativos:**
    - Escolha pelo menos 2 ativos da lista ou adicione tickers customizados
    - Os ativos devem estar disponíveis no Yahoo Finance
    
    **Período de Análise:**
    - Defina a data inicial e final para análise
    - Períodos maiores fornecem análises mais robustas
    
    **Estratégia Long & Short:**
    - Selecione qual ativo será LONG (comprado)
    - Selecione qual ativo será SHORT (vendido)
    - Escolha entre hedge ratio ótimo (calculado) ou manual
    
    **Análises Disponíveis:**
    - 📊 Estatísticas descritivas e métricas de performance
    - 📈 Evolução de preços e retornos acumulados
    - 🔗 Matriz de correlação e análise de regressão
    - 💼 Performance da estratégia Long & Short com comparações
    
    ### 🎯 O que é uma estratégia Long & Short?
    
    Uma estratégia Long & Short envolve comprar (long) um ativo que você acredita que vai subir
    e vender (short) outro ativo que você acredita que vai cair. O objetivo é lucrar com a diferença
    relativa entre os dois ativos, reduzindo a exposição ao risco de mercado.
    
    O **hedge ratio** determina a proporção entre as posições long e short, ajudando a neutralizar
    o risco de mercado e focar no spread entre os ativos.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📈 Long & Short Strategy Analyzer | Desenvolvido com Streamlit</p>
    <p style='font-size: 0.8em; color: gray;'>
        Dados fornecidos pelo Yahoo Finance | Apenas para fins educacionais
    </p>
</div>
""", unsafe_allow_html=True)
