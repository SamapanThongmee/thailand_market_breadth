# Manipulate
import datetime as dt
import pandas as pd
import numpy as np
# import math
# from scipy.stats import norm
# from datetime import datetime, timedelta

# Plot graph
# import plotly
# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.figure_factory as ff
# from plotly.subplots import make_subplots

# Finance
import yfinance as yf
# from functools import reduce

#Line
# import os
# import requests, urllib.parse
# import io
# from PIL import Image
# if not os.path.exists("images"):
    # os.mkdir("images")
    

SET_Ticker = ['2S', '3K-BAT', '7UP', 'A', 'AAI', 'AAV', 'ACC', 'ACE', 'ACG', 'ADVANC', 'AEONTS', 'AFC', 'AGE', 'AH', 'AHC', 'AI', 'AIE', 'AIMCG', 'AIMIRT', 'AIT', 'AJ', 'AJA', 'AKR', 'AKS', 'ALLA', 'ALLY', 'ALT', 'ALUCON', 'AMANAH', 'AMARIN', 'AMATA', 'AMATAR', 'AMATAV', 'AMC', 'AMR', 'ANAN', 'AOT', 'AP', 'APCO', 'APCS', 'APEX', 'APURE', 'AQUA', 'AS', 'ASAP', 'ASEFA', 'ASIA', 'ASIAN', 'ASIMAR', 'ASK', 'ASP', 'ASW', 'AURA', 'AWC', 'AYUD', 'B', 'B-WORK', 'B52', 'BA', 'BAFS', 'BAM', 'BANPU', 'BAREIT', 'BAY', 'BBGI', 'BBL', 'BCH', 'BCP', 'BCPG', 'BCT', 'BDMS', 'BEAUTY', 'BEC', 'BEM', 'BEYOND', 'BGC', 'BGRIM', 'BH', 'BIG', 'BIOTEC', 'BIZ', 'BJC', 'BJCHI', 'BKD', 'BKI', 'BKKCP', 'BLA', 'BLAND', 'BLC', 'BLISS', 'BOFFICE', 'BPP', 'BR', 'BRI', 'BROCK', 'BRR', 'BRRGIF', 'BSBM', 'BTG', 'BTNC', 'BTS', 'BTSGIF', 'BUI', 'BWG', 'BYD', 'CBG', 'CCET', 'CCP', 'CEN', 'CENTEL', 'CFRESH', 'CGD', 'CGH', 'CH', 'CHARAN', 'CHASE', 'CHAYO', 'CHG', 'CHOTI', 'CI', 'CIMBT', 'CITY', 'CIVIL', 'CK', 'CKP', 'CM', 'CMAN', 'CMC', 'CMR', 'CNT', 'COCOCO', 'COM7', 'COTTO', 'CPALL', 'CPAXT', 'CPF', 'CPH', 'CPI', 'CPL', 'CPN', 'CPNCG', 'CPNREIT', 'CPT', 'CPTGF', 'CPW', 'CRANE', 'CRC', 'CSC', 'CSP', 'CSR', 'CSS', 'CTARAF', 'CTW', 'CV', 'CWT', 'DCC', 'DCON', 'DDD', 'DELTA', 'DEMCO', 'DIF', 'DMT', 'DOHOME', 'DREIT', 'DRT', 'DTCENT', 'DTCI', 'DUSIT', 'EA', 'EASON', 'EASTW', 'ECL', 'EE', 'EGATIF', 'EGCO', 'EKH', 'EMC', 'EP', 'EPG', 'ERW', 'ERWPF', 'ESSO', 'ESTAR', 'ETC', 'EVER', 'F&D', 'FANCY', 'FE', 'FMT', 'FN', 'FNS', 'FORTH', 'FPT', 'FSX', 'FTE', 'FTI', 'FTREIT', 'FUTUREPF', 'GABLE', 'GAHREIT', 'GBX', 'GC', 'GEL', 'GENCO', 'GFPT', 'GGC', 'GIFT', 'GJS', 'GL', 'GLAND', 'GLOBAL', 'GLOCON', 'GPI', 'GPSC', 'GRAMMY', 'GRAND', 'GREEN', 'GROREIT', 'GSTEEL', 'GULF', 'GUNKUL', 'GVREIT', 'GYT', 'HANA', 'HENG', 'HFT', 'HMPRO', 'HPF', 'HTC', 'HTECH', 'HUMAN', 'HYDROGEN', 'ICC', 'ICHI', 'ICN', 'IFEC', 'IFS', 'IHL', 'III', 'ILINK', 'ILM', 'IMPACT', 'INET', 'INETREIT', 'INGRS', 'INOX', 'INSET', 'INSURE', 'INTUCH', 'IRC', 'IRPC', 'IT', 'ITC', 'ITD', 'ITEL', 'IVL', 'J', 'JAS', 'JASIF', 'JCK', 'JCT', 'JDF', 'JKN', 'JMART', 'JMT', 'JR', 'JTS', 'KAMART', 'KBANK', 'KBS', 'KBSPIF', 'KC', 'KCAR', 'KCE', 'KCG', 'KDH', 'KEX', 'KGI', 'KIAT', 'KISS', 'KKC', 'KKP', 'KPNPF', 'KSL', 'KTB', 'KTBSTMR', 'KTC', 'KTIS', 'KWC', 'KWI', 'KYE', 'L&E', 'LALIN', 'LANNA', 'LEE', 'LH', 'LHFG', 'LHHOTEL', 'LHK', 'LHPF', 'LHSC', 'LOXLEY', 'LPF', 'LPH', 'LPN', 'LRH', 'LST', 'LUXF', 'M', 'M-CHAI', 'M-II', 'M-PAT', 'M-STOR', 'MACO', 'MAJOR', 'MALEE', 'MANRIN', 'MATCH', 'MATI', 'MAX', 'MBK', 'MC', 'MCOT', 'MCS', 'MDX', 'MEGA', 'MENA', 'METCO', 'MFC', 'MFEC', 'MGC', 'MICRO', 'MIDA', 'MILL', 'MINT', 'MIPF', 'MIT', 'MJD', 'MJLF', 'MK', 'ML', 'MNIT', 'MNIT2', 'MNRF', 'MODERN', 'MONO', 'MOSHI', 'MSC', 'MST', 'MTC', 'MTI', 'NATION', 'NC', 'NCAP', 'NCH', 'NEP', 'NER', 'NEW', 'NEX', 'NFC', 'NKI', 'NNCL', 'NOBLE', 'NOK', 'NOVA', 'NRF', 'NSL', 'NTV', 'NUSA', 'NV', 'NVD', 'NWR', 'NYT', 'OCC', 'OGC', 'OHTL', 'ONEE', 'OR', 'ORI', 'OSP', 'PACE', 'PAF', 'PAP', 'PATO', 'PB', 'PCC', 'PCSGH', 'PDJ', 'PEACE', 'PERM', 'PF', 'PG', 'PHG', 'PIN', 'PJW', 'PK', 'PL', 'PLANB', 'PLAT', 'PLE', 'PLUS', 'PM', 'PMTA', 'POLAR', 'POLY', 'POPF', 'PORT', 'POST', 'PPF', 'PPP', 'PPPM', 'PQS', 'PR9', 'PRAKIT', 'PREB', 'PRECHA', 'PRG', 'PRIME', 'PRIN', 'PRINC', 'PRM', 'PRO', 'PROSPECT', 'PRTR', 'PSH', 'PSL', 'PSP', 'PT', 'PTECH', 'PTG', 'PTL', 'PTT', 'PTTEP', 'PTTGC', 'PYLON', 'Q-CON', 'QH', 'QHHR', 'QHOP', 'QHPF', 'QTC', 'RABBIT', 'RAM', 'RATCH', 'RBF', 'RCL', 'RICHY', 'RJH', 'RML', 'ROCK', 'ROH', 'ROJNA', 'RPC', 'RPH', 'RS', 'RSP', 'RT', 'S', 'S&J', 'S11', 'SA', 'SABINA', 'SABUY', 'SAK', 'SAM', 'SAMART', 'SAMCO', 'SAMTEL', 'SAPPE', 'SAT', 'SAUCE', 'SAV', 'SAWAD', 'SAWANG', 'SBNEXT', 'SC', 'SCAP', 'SCB', 'SCC', 'SCCC', 'SCG', 'SCGP', 'SCI', 'SCM', 'SCN', 'SCP', 'SDC', 'SE-ED', 'SEAFCO', 'SEAOIL', 'SENA', 'SFLEX', 'SGC', 'SGP', 'SHANG', 'SHR', 'SIAM', 'SINGER', 'SINO', 'SIRI', 'SIRIP', 'SIS', 'SISB', 'SITHAI', 'SJWD', 'SKE', 'SKN', 'SKR', 'SKY', 'SLP', 'SM', 'SMIT', 'SMK', 'SMPC', 'SMT', 'SNC', 'SNNP', 'SNP', 'SO', 'SOLAR', 'SORKON', 'SPACK', 'SPALI', 'SPC', 'SPCG', 'SPG', 'SPI', 'SPRC', 'SPRIME', 'SQ', 'SRICHA', 'SRIPANWA', 'SSC', 'SSF', 'SSP', 'SSPF', 'SSSC', 'SST', 'SSTRT', 'STA', 'STANLY', 'STARK', 'STEC', 'STECH', 'STGT', 'STHAI', 'STI', 'STPI', 'SUC', 'SUN', 'SUPER', 'SUPEREIF', 'SUSCO', 'SUTHA', 'SVI', 'SVOA', 'SVT', 'SYMC', 'SYNEX', 'SYNTEC', 'TAE', 'TAN', 'TASCO', 'TC', 'TCAP', 'TCC', 'TCJ', 'TCMC', 'TCOAT', 'TEAM', 'TEAMG', 'TEGH', 'TEKA', 'TFFIF', 'TFG', 'TFI', 'TFM', 'TFMAMA', 'TGE', 'TGH', 'TGPRO', 'TH', 'THAI', 'THANI', 'THCOM', 'THE', 'THG', 'THIP', 'THRE', 'THREL', 'TIDLOR', 'TIF1', 'TIPCO', 'TIPH', 'TISCO', 'TK', 'TKC', 'TKN', 'TKS', 'TKT', 'TLHPF', 'TLI', 'TMD', 'TMT', 'TNITY', 'TNL', 'TNPC', 'TNPF', 'TNR', 'TOA', 'TOG', 'TOP', 'TOPP', 'TPA', 'TPAC', 'TPBI', 'TPCS', 'TPIPL', 'TPIPP', 'TPOLY', 'TPP', 'TPRIME', 'TQM', 'TR', 'TRC', 'TRITN', 'TRU', 'TRUBB', 'TRUE', 'TSC', 'TSE', 'TSI', 'TSTE', 'TSTH', 'TTA', 'TTB', 'TTCL', 'TTI', 'TTLPF', 'TTT', 'TTW', 'TU', 'TU-PF', 'TVH', 'TVO', 'TWP', 'TWPC', 'TWZ', 'TYCN', 'UAC', 'UBE', 'UMI', 'UNIQ', 'UOBKH', 'UP', 'UPF', 'UPOIC', 'URBNPF', 'UTP', 'UV', 'UVAN', 'VARO', 'VGI', 'VIBHA', 'VIH', 'VNG', 'VPO', 'VRANDA', 'W', 'WACOAL', 'WAVE', 'WFX', 'WGE', 'WHA', 'WHABT', 'WHAIR', 'WHART', 'WHAUP', 'WICE', 'WIIK', 'WIN', 'WINDOW', 'WORK', 'WP', 'WPH', 'XPG', 'ZAA', 'ZEN']
tickers = [symbol + ".BK" for symbol in SET_Ticker]

batches = [tickers[i:i+50] for i in range(0, len(tickers), 50)]

def getData(ticker_list):
    start = dt.datetime.today() - dt.timedelta(days=(365*2))
    end = dt.datetime.today() + dt.timedelta(hours=7)
    high = pd.DataFrame()
    low = pd.DataFrame()
    adj_close = pd.DataFrame()

    for ticker in ticker_list:
        data = yf.download(ticker, start, end)
        high[ticker] = data['High']
        low[ticker] = data['Low']
        adj_close[ticker] = data['Adj Close']

    return high, low, adj_close

def getHLC():
    high_data = []
    low_data = []
    adj_close_data = []

    for batch in batches:
        data = getData(batch)
        high_data.append(data[0])
        low_data.append(data[1])
        adj_close_data.append(data[2])

    # Concatenate data
    concatenated_High = pd.concat(high_data, axis=1)
    concatenated_Low = pd.concat(low_data, axis=1)
    concatenated_Close = pd.concat(adj_close_data, axis=1)

    # Process data
    for df in [concatenated_High, concatenated_Low, concatenated_Close]:
        df = df[~df.index.duplicated(keep='first')]
        df = df.bfill(axis='rows')
        df = df.ffill(axis='rows')
        df = df.reset_index()

    return concatenated_High, concatenated_Low, concatenated_Close

High, Low, Close = getHLC()

column_rename_dict = {col: col.replace('.BK', '') for col in High.columns}

High.columns = High.columns.str.replace('.BK', '')
Low.columns = Low.columns.str.replace('.BK', '')
Close.columns = Close.columns.str.replace('.BK', '')

High = High.reset_index()
Low = Low.reset_index()
Close = Close.reset_index()

# New High - New Low
def getNewHighNewLow(High, Low):
    High = High.bfill().ffill()
    Low = Low.bfill().ffill()
    period_length = [20, 60, 200, 250]
    cut_rows = 250 #max(period_length)

    NewHigh = pd.DataFrame()
    NewHigh['Date'] = High['Date'].iloc[1:]
    High_without_date = High.drop(columns=['Date'])
    High_shifted = High_without_date.shift()

    number_of_stocks = High.shape[1] - 1

    for length in period_length:
        NewHigh['NH' + str(length)] = round(100 * (High_without_date > High_shifted.rolling(length).max()).sum(axis=1).iloc[1:] / number_of_stocks, 2)
    
    NewLow = pd.DataFrame()
    NewLow['Date'] = Low['Date'].iloc[1:]
    Low_without_date = Low.drop(columns=['Date'])
    Low_shifted = Low_without_date.shift()

    for length in period_length:
        NewLow['NL' + str(length)] = round(100 * (Low_without_date < Low_shifted.rolling(length).min()).sum(axis=1).iloc[1:] / number_of_stocks, 2)

    NHNL = NewHigh.merge(NewLow, on='Date', how='inner').tail(High.shape[0] - cut_rows).reset_index(drop=True)
    
    def getEMA(x, n):
        alpha = 2/(1+n)
        y[0] = x[0]
        
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1-alpha) * y[i-1]
        return y

    NHNL['DiffNHNL20d'] = getEMA((NHNL['NH20'] - NHNL['NL20']), 5)
    NHNL['DiffNHNL60d'] = getEMA((NHNL['NH60'] - NHNL['NL60']), 5)
    NHNL['DiffNHNL250d'] = getEMA((NHNL['NH250'] - NHNL['NL250']), 5)
    NHNL['DiffNHNL20d'] = round(NHNL['DiffNHNL20d'], 2)
    NHNL['DiffNHNL60d'] = round(NHNL['DiffNHNL60d'], 2)
    NHNL['DiffNHNL250d'] = round(NHNL['DiffNHNL250d'], 2)

    NHNL = NHNL.dropna().reset_index(drop=True)
    NHNL = NHNL[['Date',
                 'NH20', 'NH60', 'NH250', 'NL20', 'NL60', 'NL250',
                 'DiffNHNL20d', 'DiffNHNL60d', 'DiffNHNL250d']]
    return NHNL

# Moving Average
def getMovingAvg(Close):
    Close = Close.bfill().ffill()
    ma_length = [20, 60, 200]
    cut_rows = 200 #max(ma_length)
    
    number_of_stocks = Close.shape[1] - 1

    moving_avg_columns = ['MA' + str(length) for length in ma_length]
    moving_avg_values = pd.DataFrame()

    moving_avg_values['Date'] = Close['Date']

    for length in ma_length:
        moving_avg_values['MA' + str(length)] = round(100 * (Close.drop(columns=['Date']) > Close.drop(columns=['Date']).rolling(length).mean()).sum(axis=1) / number_of_stocks, 2)

    MovingAvg = moving_avg_values[moving_avg_columns].iloc[cut_rows:,:]
    MovingAvg['Date'] = moving_avg_values['Date']
    MovingAvg = MovingAvg[['Date', 'MA'+str(ma_length[0]), 'MA'+str(ma_length[1]), 'MA'+str(ma_length[2])]].reset_index(drop=True)
    
    return MovingAvg

def getSET():
    start = dt.datetime.today() - dt.timedelta((365*2))
    end = dt.datetime.today() + dt.timedelta(hours=7)
    SET = yf.download("^SET.BK",start,end).reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return SET

NHNL = getNewHighNewLow(High, Low)
MA = getMovingAvg(Close)
SET_Index = getSET()

SET_Index = SET_Index.merge(NHNL, on='Date', how='inner')
SET_Index = SET_Index.merge(MA, on='Date', how='inner')
print(SET_Index)