import streamlit as st
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =================================================================================
# CONFIGURAÇÃO DA PÁGINA
# =================================================================================
st.set_page_config(
    page_title="Análise Long & Short",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Sistema de Análise Long & Short")
st.markdown("""
Esta aplicação replica e moderniza a funcionalidade de um sistema de análise de pares (Long & Short) 
originalmente criado em Google Apps Script. Utilize o painel de controle à esquerda para configurar e executar a análise.
""")

# =================================================================================
# FUNÇÕES DE LÓGICA (COM TODAS AS CORREÇÕES)
# =================================================================================

@st.cache_data(ttl=3600)
def importar_dados(tickers, dias_analise):
    """
    Busca dados históricos dos ativos usando yfinance e os limpa.
    Versão robusta que lida com sufixos, a mudança 'Adj Close' -> 'Close', e falhas.
    """
    tickers_sa = [
        ticker if ticker.upper().endswith('.SA') else f"{ticker}.SA"
        for ticker in tickers
    ]
    data_final = datetime.now()
    data_inicial = data_final - timedelta(days=dias_analise)
    try:
        dados_brutos = yf.download(tickers_sa, start=data_inicial, end=data_final, progress=False)
        if dados_brutos.empty:
            st.error("A busca de dados não retornou resultados. Verifique os códigos dos ativos e o período.")
            return None, None
            
        coluna_preco = 'Close'
        if isinstance(dados_brutos.columns, pd.MultiIndex):
            if 'Close' not in dados_brutos.columns.get_level_values(0):
                if 'Adj Close' in dados_brutos.columns.get_level_values(0):
                    coluna_preco = 'Adj Close'
                else:
                    raise ValueError("Não foi possível encontrar a coluna de preços ('Close' ou 'Adj Close').")
            dados = dados_brutos[coluna_preco]
        else:
            if 'Close' in dados_brutos.columns:
                coluna_preco = 'Close'
            elif 'Adj Close' in dados_brutos.columns:
                coluna_preco = 'Adj Close'
            else:
                raise ValueError("Não foi possível encontrar a coluna de preços ('Close' ou 'Adj Close').")
            dados = dados_brutos[[coluna_preco]]
            dados.columns = [tickers[0]]
            
        if isinstance(dados, pd.Series):
            dados = dados.to_frame(name=tickers[0])
            
        colunas_renomeadas = {col: col.upper().replace('.SA', '') for col in dados.columns}
        dados.rename(columns=colunas_renomeadas, inplace=True)
        dados_validos = dados.dropna(axis=1, how='all')
        
        if dados_validos.empty:
            st.error("Todos os ativos solicitados falharam no download. Nenhum dado para analisar.")
            return None, None
            
        dados_validos.ffill(inplace=True)
        dados_validos.bfill(inplace=True)
        
        if dados_validos.isnull().values.any():
            st.warning("Alguns ativos com dados incompletos foram removidos após a limpeza.")
            dados_validos.dropna(axis=1, inplace=True)
            
        if dados_validos.empty:
            st.error("Não foi possível obter dados válidos para nenhum dos ativos fornecidos.")
            return None, None
            
        return dados_validos, (data_inicial, data_final)
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar os dados: {e}")
        st.info("Dica: Verifique se todos os tickers estão corretos e se há conexão com a internet. A API do Yahoo Finance pode estar temporariamente indisponível.")
        return None, None

def calcular_correlacao(dados_df):
    """
    Calcula a matriz de correlação entre os ativos.
    """
    return dados_df.corr()

def analisar_cointegracao(dados_df, p_value_threshold=0.05):
    """
    Realiza a análise de cointegração para todos os pares de ativos.
    """
    ativos = dados_df.columns
    resultados = []
    for i in range(len(ativos)):
        for j in range(len(ativos)):
            if i == j:
                continue
            ativo_y_ticker = ativos[i]
            ativo_x_ticker = ativos[j]
            ativo_y = dados_df[ativo_y_ticker]
            ativo_x = dados_df[ativo_x_ticker]
            modelo_x = sm.add_constant(ativo_x)
            modelo = sm.OLS(ativo_y, modelo_x).fit()
            
            # CORREÇÃO DO FUTUREWARNING: Usa .iloc para acessar por posição
            beta = modelo.params.iloc[1] 
            r_squared = modelo.rsquared
            residuos = modelo.resid
            adf_test = adfuller(residuos)
            adf_p_value = adf_test[1]
            z_score_atual = (residuos.iloc[-1] - residuos.mean()) / residuos.std()
            resultados.append({
                "Ativo Y": ativo_y_ticker,
                "Ativo X": ativo_x_ticker,
                "Beta": beta,
                "R²": r_squared,
                "ADF p-value": adf_p_value,
                "Estacionário": "Sim" if adf_p_value < p_value_threshold else "Não",
                "Z-Score Atual": z_score_atual,
                "Resíduos": residuos,
            })
    return pd.DataFrame(resultados)

def identificar_oportunidades(coint_df, correl_df, z_score_limiar):
    """
    Filtra, pontua e ranqueia as melhores oportunidades de Long & Short.
    """
    oportunidades = coint_df.copy()
    oportunidades = oportunidades[oportunidades["Estacionário"] == "Sim"]
    oportunidades = oportunidades[abs(oportunidades["Z-Score Atual"]) > z_score_limiar]
    if oportunidades.empty:
        return pd.DataFrame()
    correlacoes = []
    for index, row in oportunidades.iterrows():
        corr = correl_df.loc[row["Ativo Y"], row["Ativo X"]]
        correlacoes.append(corr)
    oportunidades["Correlação"] = correlacoes
    oportunidades["Score"] = (
        oportunidades["R²"] * 100 +
        abs(oportunidades["Z-Score Atual"]) * 50 +
        abs(oportunidades["Correlação"]) * 100
    )
    def definir_sinal(z_score):
        if z_score > z_score_limiar:
            return "Vender Y, Comprar X"
        elif z_score < -z_score_limiar:
            return "Comprar Y, Vender X"
        return "Aguardar"
    oportunidades["Sinal"] = oportunidades["Z-Score Atual"].apply(definir_sinal)
    oportunidades["Proporção (Y:X)"] = oportunidades["Beta"].apply(lambda b: f"1 : {b:.2f}")
    oportunidades.sort_values(by="Score", ascending=False, inplace=True)
    oportunidades.reset_index(drop=True, inplace=True)
    oportunidades["Rank"] = oportunidades.index + 1
    colunas_finais = [
        "Rank", "Ativo Y", "Ativo X", "Sinal", "Proporção (Y:X)", "Z-Score Atual",
        "R²", "Correlação", "ADF p-value", "Score", "Resíduos"
    ]
    return oportunidades[colunas_finais]

def estilizar_tabela_oportunidades(df):
    """Aplica formatação condicional à tabela de oportunidades."""
    return df.style.background_gradient(
        cmap='RdYlGn_r', subset=['Z-Score Atual'], vmin=-3, vmax=3
    ).background_gradient(
        cmap='viridis', subset=['R²', 'Correlação', 'Score']
    ).format({
        "Beta": "{:.2f}",
        "R²": "{:.2%}",
        "ADF p-value": "{:.4f}",
        "Z-Score Atual": "{:.2f}",
        "Correlação": "{:.2%}",
        "Score": "{:.0f}",
    })

def plotar_analise_par(dados_historicos, oportunidade_selecionada):
    """Cria um gráfico interativo para a oportunidade selecionada."""
    ativo_y = oportunidade_selecionada['Ativo Y']
    ativo_x = oportunidade_selecionada['Ativo X']
    residuos = oportunidade_selecionada['Resíduos']
    dados_normalizados = dados_historicos[[ativo_y, ativo_x]].copy()
    dados_normalizados = dados_normalizados / dados_normalizados.iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados_normalizados.index, y=dados_normalizados[ativo_y], name=f'Preço {ativo_y} (norm.)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dados_normalizados.index, y=dados_normalizados[ativo_x], name=f'Preço {ativo_x} (norm.)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=residuos.index, y=(residuos - residuos.mean()) / residuos.std(), name='Z-Score do Spread', yaxis='y2', line=dict(color='green', dash='dash')))
    z_limiar = st.session_state.z_score_limiar
    fig.add_hline(y=z_limiar, line_color="red", line_dash="dot", yref="y2", annotation_text=f'Z-Score Limiar (+{z_limiar})')
    fig.add_hline(y=-z_limiar, line_color="red", line_dash="dot", yref="y2", annotation_text=f'Z-Score Limiar (-{z_limiar})')
    fig.add_hline(y=0, line_color="grey", line_dash="dot", yref="y2", annotation_text='Média')
    fig.update_layout(
        title=f'Análise do Par: {ativo_y} vs {ativo_x}',
        xaxis_title='Data',
        yaxis_title='Preço Normalizado',
        yaxis=dict(domain=[0.3, 1]),
        yaxis2=dict(title='Z-Score do Spread', anchor='x', overlaying='y', side='right', domain=[0, 0.25]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# =================================================================================
# INTERFACE DO USUÁRIO (SIDEBAR)
# =================================================================================
st.sidebar.header("Painel de Controle")
ativos_padrao = "PETR4,VALE3,ITUB4,BBDC4,BBAS3,ITSA4,B3SA3,ABEV3"
st.sidebar.subheader("1. Lista de Ativos")
ativos_input = st.sidebar.text_area(
    "Digite os códigos dos ativos separados por vírgula:",
    value=ativos_padrao,
    height=150,
    help="Exemplo: PETR4,VALE3,ITUB4. Não é necessário incluir o sufixo '.SA'."
)
st.sidebar.subheader("2. Parâmetros de Análise")
dias_analise_input = st.sidebar.number_input(
    "Dias para Análise", min_value=30, max_value=720, value=60, step=10
)
z_score_limiar_input = st.sidebar.number_input(
    "Limiar do Z-Score para Sinal", min_value=0.5, max_value=3.0, value=1.5, step=0.1
)
st.sidebar.subheader("3. Executar")
executar_analise_btn = st.sidebar.button("Executar Análise Completa", type="primary", use_container_width=True)
if 'resultados' not in st.session_state:
    st.session_state.resultados = {}

# =================================================================================
# LÓGICA PRINCIPAL
# =================================================================================
if executar_analise_btn:
    st.session_state.z_score_limiar = z_score_limiar_input
    ativos_lista = [ativo.strip().upper() for ativo in ativos_input.split(',') if ativo.strip()]
    if len(ativos_lista) < 2:
        st.error("Por favor, insira pelo menos dois ativos para a análise.")
    else:
        with st.spinner("Executando análise completa... Este processo pode levar alguns minutos."):
            st.write("### Etapa 1: Importando Dados Históricos...")
            dados_df, periodo = importar_dados(ativos_lista, dias_analise_input)
            if dados_df is not None:
                st.session_state.resultados['dados_df'] = dados_df
                st.session_state.resultados['periodo'] = periodo
                st.success(f"Dados importados para {len(dados_df.columns)} ativos de {periodo[0].strftime('%d/%m/%Y')} a {periodo[1].strftime('%d/%m/%Y')}.")
                
                st.write("### Etapa 2: Calculando a Matriz de Correlação...")
                correl_df = calcular_correlacao(dados_df)
                st.session_state.resultados['correl_df'] = correl_df
                st.success("Matriz de correlação calculada.")
                
                st.write("### Etapa 3: Analisando a Cointegração dos Pares...")
                coint_df = analisar_cointegracao(dados_df)
                st.session_state.resultados['coint_df'] = coint_df
                st.success(f"Análise de cointegração concluída para {len(coint_df)} pares.")
                
                st.write("### Etapa 4: Identificando e Ranqueando Oportunidades...")
                oportunidades_df = identificar_oportunidades(coint_df, correl_df, z_score_limiar_input)
                st.session_state.resultados['oportunidades_df'] = oportunidades_df
                if not oportunidades_df.empty:
                    st.success(f"{len(oportunidades_df)} oportunidades encontradas!")
                else:
                    st.warning("Nenhuma oportunidade encontrada com os critérios atuais.")
                st.balloons()

# =================================================================================
# EXIBIÇÃO DOS RESULTADOS
# =================================================================================
if 'resultados' in st.session_state and st.session_state.resultados:
    st.divider()
    st.header("Resultados da Análise")
    tabs = st.tabs(["🏆 Oportunidades", "📈 Análise de Pares", "🔗 Cointegração", "🧮 Correlação", "📊 Dados Históricos"])
    
    with tabs[0]: # Oportunidades
        st.subheader("🏆 Melhores Oportunidades de Long & Short")
        oportunidades_df = st.session_state.resultados.get('oportunidades_df')
        if oportunidades_df is not None and not oportunidades_df.empty:
            df_display = oportunidades_df.drop(columns=['Resíduos'])
            st.dataframe(estilizar_tabela_oportunidades(df_display), use_container_width=True)
        else:
            st.info("Nenhuma oportunidade encontrada com os parâmetros definidos.")
            
    with tabs[1]: # Análise de Pares
        st.subheader("📈 Análise Gráfica do Par Selecionado")
        oportunidades_df = st.session_state.resultados.get('oportunidades_df')
        if oportunidades_df is not None and not oportunidades_df.empty:
            lista_pares = [f"{row['Rank']}: {row['Ativo Y']} vs {row['Ativo X']}" for index, row in oportunidades_df.iterrows()]
            par_selecionado_str = st.selectbox("Selecione uma oportunidade para visualizar o gráfico:", lista_pares)
            if par_selecionado_str:
                rank_selecionado = int(par_selecionado_str.split(':')[0])
                oportunidade_selecionada = oportunidades_df[oportunidades_df['Rank'] == rank_selecionado].iloc[0]
                dados_df = st.session_state.resultados.get('dados_df')
                fig = plotar_analise_par(dados_df, oportunidade_selecionada)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhuma oportunidade encontrada para gerar gráficos.")
            
    with tabs[2]: # Cointegração
        st.subheader("🔗 Tabela de Cointegração Completa")
        coint_df = st.session_state.resultados.get('coint_df')
        if coint_df is not None:
            df_display = coint_df.drop(columns=['Resíduos'])
            # CORREÇÃO DO VALUEERROR: Formatação seletiva
            format_dict = {
                "Beta": "{:.4f}",
                "R²": "{:.4f}",
                "ADF p-value": "{:.4f}",
                "Z-Score Atual": "{:.4f}",
            }
            st.dataframe(df_display.style.format(format_dict), use_container_width=True)
            
    with tabs[3]: # Correlação
        st.subheader("🧮 Matriz de Correlação")
        correl_df = st.session_state.resultados.get('correl_df')
        if correl_df is not None:
            st.dataframe(correl_df.style.background_gradient(cmap='coolwarm').format("{:.2%}"), use_container_width=True)
            
    with tabs[4]: # Dados Históricos
        st.subheader("📊 Dados Históricos de Fechamento Ajustado")
        dados_df = st.session_state.resultados.get('dados_df')
        if dados_df is not None:
            st.dataframe(dados_df.style.format("{:.2f}"), use_container_width=True)
else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em 'Executar Análise Completa' para começar.")

st.sidebar.divider()
st.sidebar.markdown("""
**Sobre a Análise:**
- **Cointegração:** Pares de ativos que se movem juntos a longo prazo.
- **Teste ADF:** Verifica se o spread (resíduo) entre os ativos é estacionário (reverte à média).
- **Z-Score:** Mede o quão distante o spread atual está da sua média histórica. Valores extremos (>1.5 ou <-1.5) sugerem uma oportunidade de reversão.
- **Atualização Diária:** Para automação, esta aplicação pode ser hospedada em plataformas como a Streamlit Community Cloud.
""")