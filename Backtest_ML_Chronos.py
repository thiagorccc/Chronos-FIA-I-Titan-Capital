#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import pandas_datareader as web
import pandas as pd
import datetime as dt
import numpy as np
from scipy.optimize import minimize
import pyfolio as pf
import warnings
import yfinance as yf
import matplotlib.pyplot as plt
import empyrical
from ta import momentum 
from ta import trend
from ta import volatility
from ta import volume
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta 
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
import investpy
from datetime import datetime, timedelta, date
from arch import arch_model
plt.style.use('seaborn')
print(datetime.now())

from alpha_vantage.timeseries import TimeSeries
api_alpha_vantage = 'TVVU975Z2WVV7791'
ts = TimeSeries(key = api_alpha_vantage, output_format= 'pandas')
import time
import seaborn as sns

from hurst import compute_Hc, random_walk
hurst = lambda x: compute_Hc(x)[0]

import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
yf.pdr_override()

#%% Inputs

#Restrições
bound = (0.0, .3)
rent_alvo = np.arange(-1.00, 1, 0.01)

#Data
first_price_date = dt.datetime(2014, 1, 1)
if date.today().day > 27: #No final do mês ele considera o mês como finalizado
    last_price_date = dt.datetime(2023, date.today().month + 1, 1)
else:
    last_price_date = dt.datetime(2023, date.today().month, 1)


symbols = ['PETR4.SA', 'AMER3.SA', 'ITUB4.SA','VALE3.SA','WEGE3.SA','ELET3.SA','B3SA3.SA',\
           'VIIA3.SA','COGN3.SA','OIBR3.SA','CIEL3.SA','SHOW3.SA','EMBR3.SA','CVCB3.SA',\
           'ALPA4.SA','ABEV3.SA', 'BPAN4.SA','BBSE3.SA','BBDC4.SA','BBAS3.SA',\
           'BRKM5.SA','BRFS3.SA','CCRO3.SA','CMIG4.SA','CPLE6.SA','CSAN3.SA','CPFE3.SA','CYRE3.SA',\
           'ECOR3.SA','ENEV3.SA','EGIE3.SA','EQTL3.SA','EZTC3.SA','FLRY3.SA',\
           'GGBR4.SA','GOLL4.SA','HYPE3.SA','JBSS3.SA','JHSF3.SA',\
           #'KLBN11.SA',
           'LREN3.SA','MGLU3.SA','MRFG3.SA','BEEF3.SA','MRVE3.SA','MULT3.SA','PRIO3.SA','QUAL3.SA','RADL3.SA',\
           #'RAIL3.SA',\
           'SBSP3.SA','SANB11.SA','SLCE3.SA','TAEE11.SA','VIVT3.SA',\
           'TIMS3.SA','TOTS3.SA','USIM5.SA','YDUQ3.SA']

benchmarks = ['^BVSP']

preselection = 'yes'

ml_method = 'RF'

opt_method = '1/N'

trainning_test_size = 'yearly'

test_size = 'monthly'

data_source = 'yf'

#%% Download
data = pd.DataFrame()
data_high = pd.DataFrame()
data_low = pd.DataFrame()
data_volume = pd.DataFrame()
data_benchmark = pd.DataFrame()

#Baixando os dados
#Várias formas de puxar os dados 
if data_source == 'excel':
    data =    pd.read_excel('data.xlsx').set_index('date').fillna(method='bfill').loc[first_price_date:]
    data_high =     pd.read_excel('data_high.xlsx').set_index('date').fillna(method='bfill').loc[first_price_date:]
    data_low =    pd.read_excel('data_low.xlsx').set_index('date').fillna(method='bfill').loc[first_price_date:]
    data_volume =    pd.read_excel('data_volume.xlsx').set_index('date').fillna(method='bfill').loc[first_price_date:]
else:
    for sym in tqdm(symbols):
        if data_source == 'yf':
            if sym[-3:]!='.SA':
                if sym=='BM&FBOVESPA Real Estate IFIX':
                    dados=investpy.get_index_historical_data(sym, country='brazil',\
                                                             from_date=first_price_date.strftime('%d/%m/%Y'), \
                                                             to_date=last_price_date.strftime('%d/%m/%Y')).rename(columns={'Close':'Adj Close'})
                elif sym=='Fundo de Invest Ishares SP 500':
                    dados=investpy.get_etf_historical_data(sym, country='brazil',\
                                                           from_date=first_price_date.strftime('%d/%m/%Y'),\
                                                           to_date=last_price_date.strftime('%d/%m/%Y')).rename(columns={'Close':'Adj Close'})
            else:
                dados=web.data.get_data_yahoo(sym,first_price_date,last_price_date)
            adj = dados['Adj Close']/dados['Close'] #Ajustar dados
            
        elif data_source == 'alpha':
            try:
                dados, meta_dados = ts.get_daily_adjusted(symbol = sym + 'O', outputsize = 'full')
                dados = dados.sort_index()
                dados.rename(columns={'1. open':'Open'}, inplace = True)
                dados.rename(columns={'2. high':'High'}, inplace = True)
                dados.rename(columns={'3. low':'Low'}, inplace = True)
                dados.rename(columns={'4. close':'Close'}, inplace = True)
                dados.rename(columns={'5. adjusted close':'Adj Close'}, inplace = True)
                dados.rename(columns={'6. volume':'Volume'}, inplace = True)
                dados.drop(['7. dividend amount', '8. split coefficient'], axis=1, inplace = True)
            except ValueError:
                time.sleep(60)
                dados, meta_dados = ts.get_daily_adjusted(symbol = sym + 'O', outputsize = 'full')
                dados = dados.sort_index()
                dados.rename(columns={'1. open':'Open'}, inplace = True)
                dados.rename(columns={'2. high':'High'}, inplace = True)
                dados.rename(columns={'3. low':'Low'}, inplace = True)
                dados.rename(columns={'4. close':'Close'}, inplace = True)
                dados.rename(columns={'5. adjusted close':'Adj Close'}, inplace = True)
                dados.rename(columns={'6. volume':'Volume'}, inplace = True)
                dados.drop(['7. dividend amount', '8. split coefficient'], axis=1, inplace = True)     
            adj = 1 #Dados já ajustados

        dados = dados.loc[first_price_date:last_price_date]
        data[sym] = dados['Adj Close']
        data_high[sym] = dados['High']*adj
        data_low[sym] = dados['Low']*adj
        data_volume[sym] = dados['Volume']

data_total = data
data = data.fillna(method = 'ffill')
data_high = data_high.fillna(method = 'ffill')
data_low = data_low.fillna(method = 'ffill')
data_volume = data_volume.fillna(method = 'ffill')

data_benchmark = pd.DataFrame()

returns = np.log(data / data.shift(1)) 
returns = returns.dropna()

#Baixando os benchmarks
for benchmark in benchmarks:
    data_benchmark[benchmark] = web.data.get_data_yahoo(benchmark,first_price_date,last_price_date)['Adj Close']
data_benchmark = data_benchmark.dropna()

returns_benchmark = np.log(data_benchmark / data_benchmark.shift(1)) 
returns_benchmark = returns_benchmark.dropna()

#%% Optimization 

### Markowitz with preselection ###
#if preselection == 'yes':
    
def portfolio_variance_ps(w): #matriz de covariância
    _w = np.asarray(w) #tirar o "_"
    return _w.dot(covariance_matrix).dot(_w)

def portfolio_return_ps(w):
    _w = np.asarray(w)
    return _w.dot(annualized_returns)

def portfolio_variance_gradient_ps(w):
    _w = np.asarray(w)
    return 2 * _w.dot(covariance_matrix)

def minimize_portfolio_variance_ps(min_return):
    assets_number = len(winners_list)
    bnds = [bound for i in range(assets_number)] 
    initial_guess = [1 / assets_number for i in range(assets_number)]
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},
            {'type': 'ineq', 'fun': lambda w: portfolio_return_ps(w) - min_return})
    return minimize(portfolio_variance_ps, initial_guess,
                    constraints=cons,
                    bounds=bnds,
                    options={'disp': False},
                    method='SLSQP',
                    jac=portfolio_variance_gradient_ps)

def calculate_mv_frontier_ps():
    frontier = pd.Series()
    weights = pd.DataFrame()
    for min_return in rent_alvo:
        result = minimize_portfolio_variance_ps(min_return)
        if not result.success:
            continue
        w_frontier = result.x
        port_sigma = np.sqrt(portfolio_variance_ps(w_frontier))
        port_return = portfolio_return_ps(w_frontier)
        frontier.at[port_sigma] = port_return

        for i, win in enumerate(winners_list):
            weights.at[round(port_sigma,2), win] = round(w_frontier[i],4)
            
    return frontier, weights

### Markowitz MV ###

def portfolio_variance(w):
    _w = np.asarray(w) #tirar o "_"
    return _w.dot(covariance_matrix).dot(_w)

def portfolio_return(w):
    _w = np.asarray(w)
    return _w.dot(annualized_returns)

def portfolio_variance_gradient(w):
    _w = np.asarray(w)
    return 2 * _w.dot(covariance_matrix)

def minimize_portfolio_variance(min_return):
    assets_number = len(symbols)
    bnds = [bound for i in range(assets_number)] 
    initial_guess = [1 / assets_number for i in range(assets_number)]
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},
            {'type': 'ineq', 'fun': lambda w: portfolio_return(w) - min_return})
    return minimize(portfolio_variance, initial_guess,
                    constraints=cons,
                    bounds=bnds,
                    options={'disp': False},
                    method='SLSQP',
                    jac=portfolio_variance_gradient)

def calculate_mv_frontier():
    frontier = pd.Series()
    weights = pd.DataFrame()
    for min_return in rent_alvo:
        result = minimize_portfolio_variance_ps(min_return)
        if not result.success:
            continue
        w_frontier = result.x
        port_sigma = np.sqrt(portfolio_variance_ps(w_frontier))
        port_return = portfolio_return_ps(w_frontier)
        frontier.at[port_sigma] = port_return

        for i, symbol in enumerate(symbols):
            weights.at[round(port_sigma,2), symbol] = round(w_frontier[i],4)
            
    return frontier, weights


### Hierarchical Risk Parity (HRP) ###
def calculate_hrp(returns_escolhido):
    hrp = hierarchical_risk_parity.HRPOpt(returns_escolhido)
    a=hrp.hrp_portfolio()
    a=pd.Series(a)
    a = pd.DataFrame(a)
    weights = a.T
    return weights

### 1/N ###

def calculate_ns():
    weights = pd.DataFrame(columns=symbols, index = [0])
    for sym in symbols:
        weights[sym][0] = 1.0/len(symbols)  

    return weights   

#Calculando o retorno acumulado
def portfolio_performance(initial_data_slc, end_data_slc):    
    portfolio = weights.iloc[0]
    sliced_returns = returns.loc[initial_data_slc:end_data_slc]
    retorno_ponderado = (portfolio * sliced_returns)
    retorno = retorno_ponderado.sum(axis=1)
    retorno_acumulado = (retorno+1).cumprod()
            
    return retorno, retorno_acumulado, portfolio 

def garch(serie_dados, periodos):
    
    ativo_slice=serie_dados
    retorno=np.log(ativo_slice/ativo_slice.shift(1)).dropna()
    padronizado=(retorno-retorno.mean())/retorno.std()
    
    garch_model=arch_model(retorno, p=1, q=1, vol='GARCH', dist='Normal', rescale=False)
    resultados=garch_model.fit(disp='off')
    parametros=resultados.params
    
    gama=1-parametros['alpha[1]']-parametros['beta[1]']
    vol_incondicional=retorno.std()*(252**.5)
    vol_longo_prazo=(252*parametros['omega']/gama)**.5
    
    serie_garch=pd.Series(index=retorno.index)
    serie_garch.iloc[0]=retorno.iloc[0]**2
    for i in retorno.index[1:]:
        #print(i)
        serie_garch.loc[i]=parametros['omega']+parametros['alpha[1]']*(retorno.shift(1).loc[i]**2)+parametros['beta[1]']*serie_garch.shift(1).loc[i]
            
    return ((resultados.forecast(horizon=periodos, reindex=False).variance.iloc[0])*252)**.5,(serie_garch*252)**.5

def portfolio_results(ativos, pesos, data_ini):
    
    returns_mes=pd.DataFrame()

    for i in ativos:
        returns_mes[i] = web.data.get_data_yahoo(i,data_ini,datetime(2099,12,31).date())['Adj Close']

    returns_mes=(returns_mes/returns_mes.shift(1)).fillna(1)
    returns_portfolio=np.dot(returns_mes.cumprod(),pesos)

    benchmark_mes = web.data.get_data_yahoo(benchmark,data_ini,datetime(2099,12,31).date())['Adj Close']
    ret_benchmark_mes=(benchmark_mes/benchmark_mes.shift(1)).fillna(1).cumprod()

    comparative=pd.DataFrame(index=ret_benchmark_mes.index)
    comparative.loc[:,'Strategy']=returns_portfolio
    comparative.loc[:,'Benchmark']=ret_benchmark_mes.values

    return comparative

def prints_returns():
    print('Strategy')
    print('Retorno: ',    round(np.log(returns_portfolio/returns_portfolio.shift(1))['Strategy'].dropna().mean()*252*100,2))
    print('Vol: ',    round(np.log(returns_portfolio/returns_portfolio.shift(1))['Strategy'].dropna().std()*(252**.5)*100,2))
    print('------------------')
    print('Benchmark')
    print('Retorno:',    round(np.log(returns_portfolio/returns_portfolio.shift(1))['Benchmark'].dropna().mean()*252*100,2))
    print('Vol:',    round(np.log(returns_portfolio/returns_portfolio.shift(1))['Benchmark'].dropna().std()*(252**.5),2))
    
def normalizar(df):
    return df/df.iloc[0]

def cf_analysis(cm):
    
    t=cm.iloc[0,0]+cm.iloc[0,1]+cm.iloc[1,0]+cm.iloc[1,1]
    acc_pos=cm.iloc[0,0]/(cm.iloc[0,0]+cm.iloc[0,1])
    tot_pos=(cm.iloc[0,0]+cm.iloc[1,1])/t
    acc_neg=cm.iloc[1,0]/(cm.iloc[1,0]+cm.iloc[1,1])
    tot_neg=(cm.iloc[0,1]+cm.iloc[1,0])/t
    acc_geral =(cm.iloc[0,0]+cm.iloc[1,0])/t
    
    print('')
    print('Acc Positive:',round(acc_pos,3),'x Tot Positive:',round(tot_pos,3))
    print('Acc Negative:',round(acc_neg,3),'x Tot Negative:',round(tot_neg,3))
    print('Acc Geral:',round(acc_geral,3))
    
    return 0

def MAPE(y_true, y_predict): 
    y_true, y_predict = np.array(y_true), np.array(y_predict)
    return np.mean(np.abs((y_true - y_predict) / y_true)) * 100

def analise_rrg(serie, n_days):
    #ratio entre as ações
    ratio=(100*serie)
    #macd do ratio
    macd=trend.MACD(ratio,30,10,9).macd()
    #eixo x do rrg
    eixo_x=100*(macd/ratio)
    #ema do eixo_x
    ema=trend.EMAIndicator(eixo_x, 9).ema_indicator()
    #eixo y do rrg
    eixo_y=eixo_x-ema
    
    posicao=(eixo_x,eixo_y)
    posicao_past=(eixo_x.shift(n_days), eixo_y.shift(n_days))
           
    dx = eixo_x - eixo_x.shift(n_days)
    dy = eixo_y - eixo_y.shift(n_days)
    
    posicao_pred = (eixo_x + dx, eixo_y + dy)
            
    intensidade=(dx**2+dy**2)**.5
    
    return posicao, posicao_past, posicao_pred, intensidade


data = data.fillna(method = 'bfill')
data_high = data_high.fillna(method = 'bfill')
data_low = data_low.fillna(method = 'bfill')
data_volume = data_volume.fillna(method = 'bfill')

#%% Run

#Returns Series
strategy_return = pd.Series()
bench_return = pd.Series()
naive_return = pd.Series()
looser_return = pd.Series()

#Listas para avaliar acurácia e criar matriz de confusão
predicted_values = []
actual_values = []
total_confusion_matrix = pd.DataFrame(0, index=['P','N'], columns=['T','F'])

#Listas para calcular indicadores de precisão das previsões
mse=[]
mape=[]
r_mean=[]

#Rankeando as winners/loosers para compará-las
selected_rank_winners = []
selected_rank_loosers = []

#Criando incremento das iterações
if test_size == 'monthly':
    increment = relativedelta(months=1)
elif test_size == 'quarterly':
    increment = relativedelta(months=3)
elif test_size == 'semesterly':
    increment = relativedelta(months=6)
elif test_size == 'yearly':
    increment = relativedelta(months=12)

#Criando incremento para determinar base de treino inicial
if trainning_test_size == 'monthly':
    increment_train = relativedelta(months=0)
elif trainning_test_size == 'quarterly':
    increment_train = relativedelta(months=2)
elif trainning_test_size == 'semesterly':
    increment_train = relativedelta(months=5)
elif trainning_test_size == 'yearly':
    increment_train = relativedelta(months=11)

#Data inicial do modelo
initial_data = first_price_date
#Iniciar variável da data limite de treino 
dat_aux = first_price_date + increment_train #auxiliar para iniciar o end_data
end_data = returns[(returns.index.year == dat_aux.year) & (returns.index.month == dat_aux.month)].index[-1]
    
while end_data + increment <  last_price_date:
    
    if len(strategy_return) == 0: #se for a primeira iteração entra aqui
        #data final de teste
        dat_aux = first_price_date + increment_train + increment
        end_data = returns[(returns.index.year==dat_aux.year) & (returns.index.month==dat_aux.month)].index[-1]
        #data final treino
        dat_aux = end_data - increment
        limit_train_data = returns[(returns.index.year == dat_aux.year) & (returns.index.month == dat_aux.month)].index[-1]
        
    else:
        #Nova data limite de treino
        limit_train_data = end_data
        #Nova data inicial
        dat_aux = initial_data + increment
        initial_data=returns[(returns.index.year == dat_aux.year) & (returns.index.month == dat_aux.month)].index[0] 
        #Nova data limite de teste
        dat_aux = end_data + increment
        end_data = returns[(returns.index.year == dat_aux.year) & (returns.index.month==dat_aux.month)].index[-1]
            
    #Determinando o tamanho da base de teste (tamanho total - tamanho de treino)
    size_test = len(returns.loc[initial_data:end_data]) - len(returns.loc[initial_data:limit_train_data])
    
    #Determinando o início da base de teste
    initial_test_data = returns.loc[initial_data:end_data].iloc[-size_test:].index[0]
        
    #Printar datas chave
    print('------------------------------------------------------')
    print('Início Treino: ', initial_data)
    print('Fim Treino:    ', limit_train_data)
    print('Início Teste:  ', initial_test_data)
    print('Fim Teste:     ', end_data)
        
    #Dataframe que consolida valores previstos
    prediction_consolidate = pd.DataFrame(columns = symbols)
        
    #Criar Matriz de Confusão
    confusion_matrix=pd.DataFrame(0, index=['P','N'], columns=['T','F'])
    
    for sym in symbols:
        #Slicing Database
        all_data = normalizar(data.loc[initial_data-increment:end_data])
        all_data_high = normalizar(data_high.loc[initial_data-increment:end_data])
        all_data_low = normalizar(data_low.loc[initial_data-increment:end_data])
        all_data_volume = data_volume.loc[initial_data-increment:end_data]
            
        #Creating the features dataframe
        df = pd.DataFrame()
        df['RSI'] = momentum.RSIIndicator(close=all_data[sym]).rsi()
        df['ATR'] = volatility.AverageTrueRange(high=all_data_high[sym], low=all_data_low[sym], close=all_data[sym]).average_true_range()
        df['OBV'] = volume.on_balance_volume(close=all_data[sym], volume=all_data_volume[sym])
        df['PSAR'] = trend.PSARIndicator(high=all_data_high[sym], low=all_data_low[sym], close=all_data[sym]).psar()
            
        df['EWMA 5 P'] = pd.Series.ewm(all_data[sym], span=5).mean()
        df['EWMA 10 P'] = pd.Series.ewm(all_data[sym], span=10).mean()
        df['EWMA 22 P'] = pd.Series.ewm(all_data[sym], span=22).mean()
        df['EWMA 66 P'] = pd.Series.ewm(all_data[sym], span=66).mean()
            
        df['EWMA 5 V'] = pd.Series.ewm(all_data_volume[sym], span=5).mean()
        df['EWMA 10 V'] = pd.Series.ewm(all_data_volume[sym], span=10).mean()
        df['EWMA 22 V'] = pd.Series.ewm(all_data_volume[sym], span=22).mean()
        df['EWMA 66 V'] = pd.Series.ewm(all_data_volume[sym], span=66).mean()  
            
        #Adding simple moving average for price
        df['SMA 5 P'] = all_data[sym].rolling(5).mean()
        df['SMA 10 P'] = all_data[sym].rolling(10).mean()
        df['SMA 22 P'] = all_data[sym].rolling(22).mean()
        df['SMA 66 P'] = all_data[sym].rolling(66).mean()
            
        #Adding simple moving average for volume
        df['SMA 5 V'] = all_data_volume[sym].rolling(5).mean()
        df['SMA 10 V'] = all_data_volume[sym].rolling(10).mean()
        df['SMA 22 V'] = all_data_volume[sym].rolling(22).mean()
        df['SMA 66 V'] = all_data_volume[sym].rolling(66).mean()
            
        #Adding Momentum
        df['Momentum 5'] = all_data[sym] - all_data[sym].shift(5)
        df['Momentum 10'] = all_data[sym] - all_data[sym].shift(10)
        df['Momentum 22'] = all_data[sym] - all_data[sym].shift(22)
        df['Momentum 66'] = all_data[sym] - all_data[sym].shift(66)
            
        #Adding Rolling Std Dev of prices
        df['StdDev 5 Ret'] = all_data[sym].pct_change(1).rolling(5).std()
        df['StdDev 10 Ret'] = all_data[sym].pct_change(1).rolling(10).std()
        df['StdDev 22 Ret'] = all_data[sym].pct_change(1).rolling(22).std()
        df['StdDev 66 Ret'] = all_data[sym].pct_change(1).rolling(66).std()
            
        df['Bollinger_High_2'] = volatility.BollingerBands(close = all_data[sym], window = 20, window_dev = 2).bollinger_hband() 
    
        df['Bollinger_Low_2'] = volatility.BollingerBands(close = all_data[sym], window = 20, window_dev = 2).bollinger_lband() 
    
        df['Bollinger_High_1'] = volatility.BollingerBands(close = all_data[sym], window = 20, window_dev = 1).bollinger_hband() 

        df['Bollinger_Low_1'] = volatility.BollingerBands(close = all_data[sym], window = 20, window_dev = 1).bollinger_lband()
            
        df['MACD'] = trend.MACD(close = all_data[sym], window_slow = 26, window_fast = 12, window_sign = 9).macd() 
            
        df['MACD_Signal'] = trend.MACD(close = all_data[sym], window_slow = 26, window_fast = 12, window_sign = 9).macd_signal() 
            
        df['MACD_Hist'] = trend.MACD(close = all_data[sym], window_slow = 26, window_fast = 12, window_sign = 9).macd_diff() 
            
        df['Ultimate Oscillator'] = momentum.UltimateOscillator(high = all_data_high[sym], low = all_data_low[sym], close = all_data[sym]).ultimate_oscillator() 
            
        df['Williams'] = momentum.WilliamsRIndicator(high = all_data_high[sym], low = all_data_low[sym], close = all_data[sym]).williams_r().replace(np.inf, 100).replace(-np.inf, -100)
            
        #Adding RRG
        df['RRG 5 X'] = analise_rrg(all_data[sym], 5)[0][0]
        df['RRG 10 X'] = analise_rrg(all_data[sym], 10)[0][0]
        df['RRG 22 X'] = analise_rrg(all_data[sym], 22)[0][0]
            
        df['RRG 5 Y'] = analise_rrg(all_data[sym], 5)[0][1]
        df['RRG 10 Y'] = analise_rrg(all_data[sym], 10)[0][1]
        df['RRG 22 Y'] = analise_rrg(all_data[sym], 22)[0][1]
            
        df['RRG 5 X Pred'] = analise_rrg(all_data[sym], 5)[2][0]
        df['RRG 10 X Pred'] = analise_rrg(all_data[sym], 10)[2][0]
        df['RRG 22 X Pred'] = analise_rrg(all_data[sym], 22)[2][0]
            
        df['RRG 5 Y Pred'] = analise_rrg(all_data[sym], 5)[2][1]
        df['RRG 10 Y Pred'] = analise_rrg(all_data[sym], 10)[2][1]
        df['RRG 22 Y Pred'] = analise_rrg(all_data[sym], 22)[2][1]
            
        df['RRG 5 Intensidade'] = analise_rrg(all_data[sym], 5)[3]
        df['RRG 10 Intensidade'] = analise_rrg(all_data[sym], 10)[3]
        df['RRG 22 Intensidade'] = analise_rrg(all_data[sym], 22)[3]
         
            
        df = df.loc[initial_data:end_data] #Slice final
        df_total = df #Salvar DataFrame 
        df = df.dropna() 
        
        #Retornos Diários
        retornos = np.log(all_data[sym]/all_data[sym].shift(1))  
        retornos = retornos.dropna()
        
        #Retornos após período de size_test dias
        retornos_periodo = np.log(all_data[sym]/all_data[sym].shift(size_test))
        retornos_periodo = retornos_periodo.dropna()
        
        #Criar modelo de normalização
        scaler = StandardScaler()
        scaler.fit(df.shift(size_test).dropna())
        
        #Criar Dataframe normalizado
        df_normalizado = scaler.transform(df.shift(size_test).dropna())
        df_normalizado = pd.DataFrame(df_normalizado, index = df.shift(size_test).dropna().index, columns = df.columns)
            
        #Cria Dataframe com valores binários
        #se após um período size_test dias subiu, 1, se não, -1
        df_trend_deterministic = pd.DataFrame(index = df_normalizado.index, columns = df_normalizado.columns)
        for col in df_trend_deterministic.columns:
            df_trend_deterministic.loc[:,col] = np.where(df_normalizado[col].pct_change(size_test)>0, 1,-1)
            
        #Cria X_train, X_test, y_train, y_test
        X = df_trend_deterministic
        y_reg = pd.merge(df, retornos_periodo, how='inner', on = 'Date')[sym].iloc[size_test:]
        y_bin = np.where(y_reg > 0 , 1, -1)
        y_df = y_reg.copy()
        y = pd.DataFrame(y_bin, index = y_reg.index).iloc[:,0]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_test, shuffle=False)
            
        #Determinar modelo ML
        if ml_method == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5)
        if ml_method == 'RF':
            model = RandomForestClassifier(n_estimators = 100, random_state = 0)
        if ml_method == 'SVR':
            try:
                model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            except ValueError:
                model = RandomForestClassifier(n_estimators = 100, random_state = 0)
        
        #Trienar ML e prever resultados
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        #Guardar MSE, MAPE e previsão média
        mse.append(mean_squared_error(y_test+1, y_pred+1))
        mape.append(MAPE(y_test+1, y_pred+1))
        r_mean.append(y_pred.mean())

        #Registrando a previsão 
        prediction = pd.DataFrame(data=y_pred,columns=[sym])
        prediction_consolidate[sym] = prediction[sym]
            
        #creating confusion matrix     
        if prediction[sym].sum() > 0:
            if (1+retornos.loc[initial_test_data:]).cumprod().iloc[-1] > 1:
                confusion_matrix.loc['P','T'] = confusion_matrix.loc['P','T'] + 1
            else:
                confusion_matrix.loc['P','F'] = confusion_matrix.loc['P','F'] + 1
        elif prediction[sym].sum() < 1:
            if (1+retornos.loc[initial_test_data:]).cumprod().iloc[-1] < 1:
                confusion_matrix.loc['N','T'] = confusion_matrix.loc['N','T'] + 1
            else:
                confusion_matrix.loc['N','F'] = confusion_matrix.loc['N','F'] + 1
        
        #Guardar valores previstos e realizados
        predicted_values.append((prediction[sym]).iloc[-1])
        actual_values.append((retornos.loc[initial_test_data:]).cumprod().iloc[-1])
                    
    #Adicionar dados da matriz de confusão á amtriz consolidada
    total_confusion_matrix = total_confusion_matrix + confusion_matrix
            
    #Calculando o retorno acumulado
    cummulative_ret_pred = prediction_consolidate.sum()
        
    #Selecionando os ativos com retorno positivo
    winners_df = pd.DataFrame(columns=['Rent'])
    winners_df['Rent'] = cummulative_ret_pred
    total_df = winners_df.copy() #Salvar winners_df original para fazer loosers
    winners_df = winners_df.nlargest(10,'Rent')
    winners_list = list(winners_df.index)
    
    #Criar inputs da otimização
    winners_ret = returns.filter(items=winners_list, axis=1)
    winners_ret =  winners_ret.loc[y_train.index[0]:y_train.index[-1]]  
    covariance_matrix = np.asarray(252 * winners_ret.cov())
    annualized_returns = np.asarray(252 * winners_ret.mean())
        
    #Criando ranking dos retornos para ver se as escolhidas foram bem no relativo 
    ranking = all_data.pct_change(size_test).iloc[-1].rank(ascending = False).loc[winners_df.index]
    selected_rank_winners.append((ranking-.5).mean())
                
    #Ponderação das carteiras       
    if preselection == 'yes':
        if opt_method == 'MV':
            frontier, weights = calculate_mv_frontier_ps()
        if opt_method == 'HRP':
            weights = calculate_hrp(winners_ret)
        if opt_method == '1/N':    
            winners_list = list(winners_ret.columns)
            weights = pd.DataFrame(columns = winners_list)
            weights.loc[0] = [1 / len(winners_list) for i in range(len(winners_list))]
    else:
        if opt_method == 'MV':
            frontier, weights = calculate_mv_frontier()
        if opt_method == 'HRP':
            weights = calculate_hrp(returns)
        if opt_method == '1/N':
            weights = calculate_ns()
        
    #Resultados da Carteira Winners
    retorno, retorno_acumulado, portfolio = portfolio_performance(initial_test_data, end_data)
    strategy_return = pd.concat([strategy_return, retorno])
        
    print('')
    print('Portfolio:')
    print(portfolio)
    print('')
    
    #Resultados do Benchmark
    benchmark_rets = returns_benchmark.loc[initial_data:end_data]
    bench_return = pd.concat([bench_return, benchmark_rets.loc[initial_test_data:end_data].squeeze()])
        
    #Criando carteira Naive
    winners_list = list(all_data.columns)
    weights = pd.DataFrame(columns = winners_list)
    weights.loc[0] = [1 / len(winners_list) for i in range(len(winners_list))]
       
    #Resultados da Carteira Naive
    retorno_naive, retorno_acumulado_naive, portfolio_naive = portfolio_performance(initial_test_data,end_data)
    naive_return = pd.concat([naive_return, retorno_naive])
        
    #Criando carteira Looser
    winners_df = total_df.nsmallest(10,'Rent')
    winners_list = list(winners_df.index)    
    winners_ret = returns.filter(items=winners_list, axis=1)
    winners_ret =  winners_ret.loc[y_train.index[0]:y_train.index[-1]]
    winners_list = list(winners_ret.columns)
    weights = pd.DataFrame(columns = winners_list)
    weights.loc[0] = [1 / len(winners_list) for i in range(len(winners_list))]
    
    #Resultados da Carteira Looser
    retorno_looser, retorno_acumulado_looser, portfolio_looser = portfolio_performance(y_test.index[0],y_test.index[-1])
    looser_return = pd.concat([looser_return, retorno_looser])
    
    #Criando ranking dos retornos para ver se as escolhidas foram bem no relativo
    ranking = all_data.pct_change(size_test).iloc[-1].rank(ascending = False).loc[winners_df.index]
    selected_rank_loosers.append((ranking-.5).mean())
        
    #Prints e mais prints
    print('Janela Treino:         ', len(returns.loc[initial_data:end_data].loc[initial_data:limit_train_data])-    len(returns.loc[initial_data:end_data].loc[initial_test_data:end_data]))
    print('Janela Teste:        ', len(returns.loc[initial_data:end_data].loc[initial_test_data:end_data]))
    print('Retorno:              ', (retorno+1).cumprod()[-1]-1)
    print('Retorno Looser:              ', (retorno_looser+1).cumprod()[-1]-1)
    print('Retorno Benchmark:    ',     (benchmark_rets.loc[initial_test_data:end_data]+1).cumprod().iloc[-1].values[0]-1)
    print('Retornos Acumulados:  ',    (strategy_return+1).cumprod()[-1],'x',(bench_return+1).cumprod().iloc[-1], 'x',    (naive_return+1).cumprod().iloc[-1], 'x',(looser_return+1).cumprod().iloc[-1] )
    print('Volatilidade:         ', np.sqrt(portfolio_variance(portfolio)))
    print('Sharpe Ratio:         ', empyrical.sharpe_ratio(returns=retorno))
    print('Maximum Draw Down:    ', empyrical.max_drawdown(returns=retorno))
    print('Value at Risk:        ', empyrical.value_at_risk(returns=retorno, cutoff=0.01))
    #print('Beta:                 ',\
    #      empyrical.beta(returns=retorno, factor_returns=benchmark_rets[benchmarks])) 
    #print('------------------------------------------------------')
    print('')
    print(confusion_matrix)
    cf_analysis(confusion_matrix)
    print('')
    print(total_confusion_matrix)
    cf_analysis(total_confusion_matrix)
    print('')
    print('MSE:',round(100*sum(mse)/len(mse),3),'%')
    print('MAPE:',round(sum(mape)/len(mape),3),'%')
    print('Retorno Médio:',round(100*sum(r_mean)/len(r_mean),3),'%')
    print('Ranking Winners:',round(sum(selected_rank_winners)/len(selected_rank_winners),3))
    print('Ranking Loosers:',round(sum(selected_rank_loosers)/len(selected_rank_loosers),3))
    print('')

        
#criando incremento para nova data inicial
dat_aux = initial_data + increment
new_initial_data = returns[(returns.index.year == dat_aux.year) & (returns.index.month == dat_aux.month)].index[0] 

cf_analysis(total_confusion_matrix)

rankings = pd.DataFrame(index=range(len(selected_rank_winners)),                                       columns=['Winners', 'Loosers', 'Naive'])

rankings['Winners'] = selected_rank_winners
#comparative['Benchmark'] = bench_return
rankings['Naive'] = [31.5 for i in range(len(selected_rank_winners))]
rankings['Loosers'] = selected_rank_loosers

rankings.plot()

(rankings['Loosers'] - rankings['Winners']).rolling(12).mean().plot()

#Montando a planilha de resultados

analysis_sheet = pd.DataFrame(data = strategy_return)
analysis_sheet['Return'] = empyrical.annual_return(returns=strategy_return)
analysis_sheet['Volatility'] = empyrical.annual_volatility(returns=strategy_return)
analysis_sheet['Sharpe Ratio'] = empyrical.sharpe_ratio(returns=strategy_return)
analysis_sheet['Max DD'] = empyrical.max_drawdown(returns=strategy_return)
analysis_sheet['CVaR'] = empyrical.stats.conditional_value_at_risk(returns=strategy_return, cutoff= 0.05)
#analysis_sheet['Beta'] = empyrical.beta(returns=retorno, factor_returns=benchmark_rets['^BVSP'])
analysis_sheet['retorno ac.'] = (strategy_return+1).cumprod()
#analysis_sheet['N ativos médio'] = np.mean(n_ativos)
  
analysis_sheet.to_excel('Resultados/BR/'+ml_method + '-' + opt_method + '-' + trainning_test_size + '-' + test_size + '_'+data_source+'.xlsx', index = True)

#%% Graphs
#Drawdown
fig, ax1 = plt.subplots(figsize=(22,16))
drawdown_fig=pf.plot_drawdown_periods(returns=strategy_return, label= 'Drawdown')
drawdown_fig=pf.plot_drawdown_periods(returns=strategy_return)
plt.title('Top 10 Drawdown', fontsize=22)
plt.ylabel('Retorno', fontsize=18)
plt.legend(fontsize=18, loc=2)
drawdown_fig.figure.savefig('Drawdown.png', format = 'png')
    
#Volatility
fig, ax1 = plt.subplots(figsize=(22,16))
rolling_volatility_fig=pf.plot_rolling_volatility(returns=strategy_return,                    factor_returns=returns_benchmark['^BVSP'][retorno.index[0]:retorno.index[-1]])
plt.title('Rolling Volatility', fontsize=22)
plt.ylabel('Volatilidade', fontsize=18)
plt.legend(fontsize=18, loc=2)
rolling_volatility_fig.figure.savefig('Rolling Volatility.png', format = 'png')
    
#Cumulative Returns
fig, ax1 = plt.subplots(figsize=(22,16))
plt.plot((1+strategy_return).cumprod(), 'b', label='Portfolio')
plt.plot((1+returns_benchmark['^BVSP'][strategy_return.index[0]:strategy_return.index[-1]]).cumprod(),         'r', label='^BVSP')
plt.title('Cumulative Returns', fontsize=22)
plt.ylabel('Retorno', fontsize=18)
plt.legend(fontsize=18, loc=2)
fig.savefig('Retorno Acumulado.png', format = 'png')

#Não estou conseguindo funcionar com benchmarks, vou deixar comentado

#pf.create_full_tear_sheet(strategy_return, benchmark_rets=bench_return)
#pf.create_full_tear_sheet(strategy_return, benchmark_rets=naive_return)
#pf.create_full_tear_sheet(strategy_return, benchmark_rets=looser_return)
pf.create_full_tear_sheet(strategy_return)

comparative = pd.DataFrame(index=strategy_return.index, columns=['Strategy', 'Benchmark', 'Naive', 'Looser'])

comparative['Strategy'] = strategy_return
comparative['Benchmark'] = bench_return
comparative['Naive'] = naive_return
comparative['Looser'] = looser_return

(1+comparative).cumprod().plot()

#Comparando com Looser
(((1+comparative).cumprod()['Strategy']-(1+comparative).cumprod()['Looser'])-1).plot()

#Comparando com Naive
((1+comparative).cumprod()['Strategy']-(1+comparative).cumprod()['Naive']).plot()

#Comparando com Benchmark
((1+comparative).cumprod()['Strategy']-(1+comparative).cumprod()['Benchmark']).plot()

#Mapa de calor das Correlações
sns.heatmap(comparative.corr(), annot=True, cmap='coolwarm')

#Dias que a estratégia ficou acima do Benchmark
comparative['Strategy > Benchmark'] = np.where(comparative['Strategy'] > comparative['Benchmark'], 1, 0)

print('A estratégia superou o benchmark',       round((comparative['Strategy > Benchmark'].sum()/len(comparative))*100, 2),'% dos dias')

comparative.drop('Strategy > Benchmark', axis=1, inplace=True)


#Mais algumas visualizações de estatísticas descritivas
sns.distplot(comparative['Strategy'], title = 'Strategy')

sns.distplot(comparative['Benchmark'], title = 'Benchmark')

sns.distplot(comparative['Naive'], title = 'Naive')

sns.boxplot(x = 'Strategy', data = comparative, title = 'Strategy')

sns.boxplot(x = 'Benchmark', data = comparative, title = 'Benchmark')

sns.boxplot(x = 'Naive', data = comparative, title = 'Naive')

sns.pairplot(comparative, title = 'Comparativo Diário')

#Criando comparativos mensais
comparative['Month'] = pd.Series(str)
for i in comparative.index:
    if i.month<10:
        comparative.loc[i,'Month'] = str(i.year)+'-0'+str(i.month)
    else:
        comparative.loc[i,'Month'] = str(i.year)+'-'+str(i.month)

comparative_month = comparative.copy()

comparative_month.loc[:,'Strategy'] = comparative_month.loc[:,'Strategy'] + 1
comparative_month.loc[:,'Benchmark'] = comparative_month.loc[:,'Benchmark'] + 1
comparative_month.loc[:,'Naive'] = comparative_month.loc[:,'Naive'] + 1

comparative_month = comparative_month.groupby('Month').prod() - 1 

sns.distplot(comparative_month['Strategy'], title = 'Strategy Monthly')

sns.distplot(comparative_month['Benchmark'], title = 'Benchmark Monthly')

sns.distplot(comparative_month['Naive'], title = 'Naive Monthly')

sns.boxplot(x = 'Strategy', data = comparative_month, title = 'Strategy Monthly')

sns.boxplot(x = 'Benchmark', data = comparative_month, title = 'Benchmark Monthly')

sns.boxplot(x = 'Naive', data = comparative_month, title = 'Naive Monthly')

sns.pairplot(comparative_month, title = 'Comparativo Mensal')

