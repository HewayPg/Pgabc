import sys
import pandas as pd
import time
import datetime
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import talib as ta


class Indicator():
    """数据指数函数 可变列host_nv ,user_wm,passwd_wm,db_wm,chareset_wm来适应不同要求"""

    period = 6    #系数
    weight = 0.2
    k = 12
    ts_code = '000001.SZ'
    time_temp = datetime.datetime.now() - datetime.timedelta(days=0)
    given_dt = time_temp.strftime('%Y%m%d')

    def __init__(self,df_wm,start_dt,end_dt):   
        self.df_wm = df_wm
        self.start_dt = start_dt
        self.end_dt = end_dt
        #self.cho_dt = self.cho_dt

    def db_ts_code(self):
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        return data
    """
    def ma(self):
        #返回是的DATA 可以用data.iloc[-1].ma_5等方式获得数据,可以直接用ma_wm[1]来得到最后一个数据
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        data['ma_{}'.format(str(self.k))] = data.close.rolling(self.k).mean()    #5日收盘价均值
        print(data.tail())
        ma_wm = data.iloc[-1]['ma_{}'.format(str(self.k))]
        return data,ma_wm
    """
    def ma(self,k:int = 10):
        #index_1
        #返回是的DATA 可以用data.iloc[-1].ma_5等方式获得数据,可以直接用ma_wm[1]来得到最后一个数据
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        data['ma_{}'.format(str(k))] = data.close.rolling(k).mean()    #5日收盘价均值
        #print(data.tail())
        #ma_wm = data.iloc[-1]['ma_{}'.format(str(k))]
        return data



    def ema(self,k:int = 10):
        #index_2
        #返回是的DATA 可以用data.iloc[-1].ma_5等方式获得数据,可以直接用ma_wm[1]来得到最后一个数据
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型        
        for i in range(len(data)):
            if i==0:
                data.ix[i,'ma_{}'.format(str(k))]=data.ix[i,'close']
            if i>0:
                data.ix[i,'ma_{}'.format(str(k))]=(2*data.ix[i,'close']+(k-1)*data.ix[i-1,'ma_{}'.format(str(k))])/(k+1)        
        return data


    def rsi_wm(self,period:int = 6):
        #index_3  不推荐
        #RSI相对强弱指标
        #相对强弱指数（RSI）是通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力，从而作出未来市场的走势
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        price = data.close
        
        p_change = price - price.shift(1)
        p_change = p_change.dropna()

        p_index = p_change.index
        p_up = pd.Series(0,index=p_index)
        p_up[p_change>0]=p_change[p_change>0]

        p_down = pd.Series(0,index=p_index)
        p_down[p_change<0]=-p_change[p_change<0]

        rsi_data = pd.concat([price,p_change, p_up, p_down],axis=0)
        #rsi_data = pd.concat([p_close,p_up,p_down],ignore_index=False)
        print(rsi_data)
        rsi_data.columns = ['price', 'p_change', 'p_up', 'p_down']
        rsi_data = rsi_data.dropna()

        sum_up = []  #统计UP数量
        sum_down = [] #统计DOWN数量
        for i in range(period,len(p_up)+1):
            sum_up.append(np.mean(p_up.values[(i-period):i],dtype=np.float32))
            sum_down.append(np.mean(p_down.values[(i-period):i],dtype=np.float32))
            rsi_wm = [100*sum_up[i]/(sum_up[i] + sum_down[i]) for i in range(0,len(sum_up))]  #计算RSI 6日的值

        index_rsi = p_index[period-1:]
        rsi_wm = pd.Series(rsi_wm,index=index_rsi)
        return rsi_wm

    
    def sma(self,k:int = 5):
        #index_4
        #计算PRICE_WM K天的简单平均数
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        price_wm = data.close

        sma = pd.Series(0.0,index=price_wm.index)
        for i in range(len(price_wm)):
            sma[i] = sum(price_wm[(i-k+1):(i+1)])/k
        return sma



    
    def wma_wm(self):
        #index_5  不推荐
        # 计算PRICE_WM weight权重的的加权平均数
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        price_wm = data.close

       
        b = np.array([1, 2, 3, 4, 5]) #计算权重   放在函数的前面
        weight = b / sum(b)
        #print(weight)
       
        k = len(weight)
        arrw = np.array(weight)
        arrw = 1
        wma = pd.Series(0.0,index=price_wm.index)
        for i in range(k-1,len(price_wm.index)):
            wma[i] = sum(arrw*price_wm[(i-k+1):(i+1)])

        #data['wma'] = wma
        return wma


    def wma(self,timeperiod:int=30):
        #index_5 wma 
        #WMA - Weighted Moving Average
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['wma'] = ta.WMA(data.close,timeperiod=timeperiod)
        return data
 

    
    
    def ewma(self,period:int = 5,exponential:float = 0.2):
        #index_6
        # 计算PRICE_WM 期数为PERIOD天,EXPONENTIAL权数的指数加权平均数
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        price_wm = data.close

        ewma = pd.Series(0.0,index=price_wm.index)
        ewma[period-1]=np.mean(price_wm[:period])
        for i in range(period,len(price_wm)):
            ewma[i]=exponential*price_wm[i]+(1-exponential)*ewma[i-1]

        data['ewma'] = ewma
        return data



    
    def momentum(self,period:int = 19):
        #index_7
        #momentum trading strategy 动量交易策略 现价比PERIOND_WM前的价差
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        price_wm = data.close

        lagprice = price_wm.shift(period)
        momen = price_wm - lagprice
        #momen = momen.dropna()
        data['momentum'] = momen
        return data



    def macd(self,period1:int = 12,period2:int = 26,period3:int=9):
        #index_8
       #macd 12,26,9 数据
       data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
       data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
       data = data.reset_index()   #重新排
       item_variable = 'close'   #设定转换类型的变更
       data[item_variable] = data[item_variable].astype('float')  #转换数据类型
       data['dif']=data['close'].ewm(adjust=False,alpha=2/(period1+1),ignore_na=True).mean()-data['close'].ewm(adjust=False,alpha=2/(period2+1),ignore_na=True).mean()
       data['dea']=data['dif'].ewm(adjust=False,alpha=2/(period3+1),ignore_na=True).mean()
       data['macd']=2*(data['dif']-data['dea'])
       return data



    def kdj(self,n:int = 9,m1:int = 3,m2:int = 3):
        #index_9
        #kdj 9,3,3 数据
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        low_list = data['low'].rolling(n, min_periods=n).min()
        low_list.fillna(value=data['low'].expanding().min(), inplace=True)
        high_list = data['high'].rolling(n, min_periods=n).max()
        high_list.fillna(value = data['high'].expanding().max(), inplace=True)
        rsv = (data['close'] - low_list) / (high_list - low_list) * 100
        data['k'] = pd.DataFrame(rsv).ewm(com=2).mean()
        data['d'] = data['k'].ewm(com=2).mean()
        data['j'] = 3 * data['k'] - 2 * data['d']
        return data

    def bias(self,n1:int = 6,n2:int = 12,n3:int = 24):
        #index_10
        #bias指标  6,12,24
        #N期BIAS=(当日收盘价-N期平均收盘价)/N期平均收盘价*100%
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['bias_{}'.format(str(n1))] = (data['close'] - data['close'].rolling(n1, min_periods=1).mean())/ data['close'].rolling(n1, min_periods=1).mean()*100
        data['bias_{}'.format(str(n2))] = (data['close'] - data['close'].rolling(n2, min_periods=1).mean())/ data['close'].rolling(n2, min_periods=1).mean()*100
        data['bias_{}'.format(str(n3))] = (data['close'] - data['close'].rolling(n3, min_periods=1).mean())/ data['close'].rolling(n3, min_periods=1).mean()*100
        data['bias_{}'.format(str(n1))] = round(data['bias_{}'.format(str(n1))], 2)
        data['bias_{}'.format(str(n2))] = round(data['bias_{}'.format(str(n2))], 2)
        data['bias_{}'.format(str(n3))] = round(data['bias_{}'.format(str(n3))], 2)
        return data

    def wr(self,n1:int = 10,n2:int = 6):
        #index_11
        #威廉指标
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['wr1'] =  0 - ta.WILLR(data['high'], data['low'], data['close'], timeperiod=n1)
        data['wr2'] =  0 - ta.WILLR(data['high'], data['low'], data['close'], timeperiod=n2)
        return data

    def obv(self,n:int = 30):
        #index_12
        #股市技术分析的四大要素：价、量、时、空。OBV指标就是从“量”这个要素作为突破口，来发现热门股票、分析股价运动趋势的一种技术指标
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'vol'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['obv'] = ta.OBV(data.close, data.vol)
        data['obv_{}'.format(str(n))] = data.obv.rolling(n).mean()    #5日收盘价均值
        return data

    def boll(self):
        #index_13 
        #布林线指标，即BOLL指标，其英文全称是“Bollinger Bands”
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['upper'], data['middle'], data['lower'] = ta.BBANDS(data.close.values,timeperiod=20,nbdevup=2,nbdevdn=2,matype=0)
        return data


    def sun_moon(self):
        #index_14
        #生成阳线阴线标志，阳线为True ,阴线为False        
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        wm = lambda x,y : x > y        #判断是否收阳线     
        data['sun_moon'] = wm(data['close'],data['open'])
        return data

    def k_form(self):
        #index_15
        #生成K线实体占HIGH-LOW的比重*100
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['co_hl'] = 0.000001
        data['co_hl'] = data.apply(lambda x: (x['close']-x['open'])/(x['high']-x['low'])*100 if x['high']-x['low'] !=0 else 0, axis=1)

        #k线生成上影线和下影线占整体比重*100
        def head_total(o,h,l,c):
            #上影线占整体比*100
            k_wm = [o,h,l,c]
            k_wm.sort()
            if k_wm[0] == k_wm[3]:
                return 0
            else:
                return (k_wm[3]-k_wm[2])/(k_wm[3] - k_wm[0])*100

        def tail_total(o,h,l,c):
            #下影线占整体比*100
            k_wm = [o,h,l,c]
            k_wm.sort()
            if k_wm[0] == k_wm[3]:
                return 0
            else:
                return (k_wm[1]-k_wm[0])/(k_wm[3] - k_wm[0])*100
            
        data['head_total'] = 0.000001
        data['tail_total'] = 0.000001
        data['head_total'] = data.apply(lambda x:head_total(x['open'],x['high'],x['low'],x['close']),axis=1)
        data['tail_total'] = data.apply(lambda x:tail_total(x['open'],x['high'],x['low'],x['close']),axis=1)
        return data


    def ma_group(self):
        #index_16 ma
        #生成数据的
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'vol'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data =data[data[item_variable].notnull() & data[item_variable] != 0]  #删除N_INCOME TOTAL_REVENUE为空和零的值

        data['ma_5'] = data.close.rolling(5).mean()    #5日收盘价均值
        data['ma_10'] = data.close.rolling(10).mean()    #5日收盘价均值
        data['ma_20'] = data.close.rolling(20).mean()    #5日收盘价均值
        data['ma_30'] = data.close.rolling(30).mean()    #5日收盘价均值
        data['ma_60'] = data.close.rolling(60).mean()    #5日收盘价均值
        data['ma_120'] = data.close.rolling(120).mean()    #5日收盘价均值
        data['ma_250'] = data.close.rolling(250).mean()    #5日收盘价均值
        data['ma_500'] = data.close.rolling(500).mean()    #5日收盘价均值    

        data['vol_5'] = data.vol.rolling(5).mean()    #5日收盘价均值
        data['vol_10'] = data.vol.rolling(10).mean()    #5日收盘价均值
        data['vol_20'] = data.vol.rolling(20).mean()    #5日收盘价均值
        data['vol_30'] = data.vol.rolling(30).mean()    #5日收盘价均值
        data['vol_60'] = data.vol.rolling(60).mean()    #5日收盘价均值
        data['vol_120'] = data.vol.rolling(120).mean()    #5日收盘价均值
        data['vol_250'] = data.vol.rolling(250).mean()    #5日收盘价均值
        data['vol_500'] = data.vol.rolling(500).mean()    #5日收盘价均值

        return data

    def sma_ta(self,timeperiod:int = 30):
        #index_17 sma
        #SMA （简单移动平均线）（参数1：收盘价序列，参数2：时间周期（均线的计算长度 即 几日均线））
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['sma'] = ta.SMA(data.close, timeperiod=timeperiod)
        return data

    def atr(self,timeperiod:int = 14):
        #index_18
        #ATR（平均真实波幅）
        #（参数1：最高价序列，参数2：最低价序列，参数3：收盘价序列，参数4：时间周期）
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['atr'] = ta.ATR(data.high, data.low, data.close, timeperiod=14)
        return data

    def ht_trendline(self):
        #inex_19 ht_trendline
        #HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['ht_trendline'] = ta.HT_TRENDLINE(data.close)
        return data

    def kama(self,timeperiod:int=30):
        #index_20 kama
        #KAMA - Kaufman Adaptive Moving Average
        #NOTE: The KAMA function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['kama'] = ta.KAMA(data.close, timeperiod=timeperiod)
        return data

    def var(self,timeperiod:int = 5,nbdev:int = 1):
        #index_21 var
        #MAMA - MESA Adaptive Moving Average
        #NOTE: The MAMA function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['var'] = ta.VAR(data.close, timeperiod=timeperiod, nbdev=nbdev)       
        return data

    def tsf(self,timeperiod:int=14):
        #index_22 var
        #TSF - Time Series Forecast
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
       
        data['tsf'] = ta.TSF(data.close, timeperiod=timeperiod)
        return data

    def dema(self,timeperiod:int = 30):
        #index_23 dema
        #DEMA - Double Exponential Moving Average
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['dema'] = ta.DEMA(data.close, timeperiod=timeperiod)
        return data

    def mama(self):
        #not successful !!!!!!!!!!!!!
        #index_24 mama
        #MAMA - MESA Adaptive Moving Average
        #mama, fama = MAMA(close, fastlimit=0, slowlimit=0)
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['mama'],data['fama'] = ta.MAMA(data.close, fastlimit=0, slowlimit=0)
        return data

    def mavp(self,periods:int = 12,minperiod:int=2,maxperiod:int=30):
        #not successful !!!!!!!!!!!!!
        #index_25 mavp
        #MAVP - Moving average with variable period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['mavp'] = ta.MAVP(data.close, periods, minperiod=minperiod, maxperiod=maxperiod, matype=0)
        return data

    def midpoint(self,timeperiod:int=14):
        #MIDPOINT - MidPoint over period
        #index_26 minpoint
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['midpoint'] = ta.MIDPOINT(data.close, timeperiod=timeperiod)
        return data

    def midprice(self,timeperiod:int=14):
        #index_27 midprice
        #MIDPRICE - Midpoint Price over period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['midprice'] = ta.MIDPRICE(data.high, data.low, timeperiod=timeperiod)
        return data

    def sar(self):
        #index_28 sar
        #SAR - Parabolic SAR
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['sar'] = ta.SAR(data.high, data.low, acceleration=0, maximum=0)
        return data

    def sarext(self):
        #index_29 sarext
        #SAREXT - Parabolic SAR - Extended
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['sarext'] = ta.SAREXT(data.high, data.low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
        return data

    def t3(self,timeperiod:int = 5):
        #index_30 t3
        #T3 - Triple Exponential Moving Average (T3)
        #NOTE: The T3 function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['t3'] = ta.T3(data.close, timeperiod=timeperiod, vfactor=0)
        return data


    def tema(self,timeperiod:int=30):
        #index_31 tema
        #TEMA - Triple Exponential Moving Average
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['tema'] = ta.TEMA(data.close, timeperiod=timeperiod)
        return data

    def trima(self,timeperiod:int=30):
        #index_32 tirma
        #TRIMA - Triangular Moving Average
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['trima'] = ta.TRIMA(data.close, timeperiod=timeperiod)
        return data

    def wma1(self,timeperiod:int = 30):
        #index_33 wma
        #WMA - Weighted Moving Average
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['wma'] = ta.WMA(data.close, timeperiod=timeperiod)
        return data

    def adx(self,timeperiod:int = 14):
        #index_34 adx
        #ADX - Average Directional Movement Index
        #NOTE: The ADX function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['adx'] = ta.ADX(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def adxr(self,timeperiod:int=14):
        #index_35 adxr
        #ADXR - Average Directional Movement Index Rating
        #NOTE: The ADXR function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['adxr'] = ta.ADXR(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def apo(self,fastperiod:int=12,slowperiod:int=26):
        #index_36 apo
        #APO - Absolute Price Oscillator
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['apo'] = ta.APO(data.close, fastperiod=fastperiod, slowperiod=slowperiod, matype=0)
        return data

    def aroon(self,timeperiod:int=14):
        #index_37 aroon
        #AROON - Aroon
        #aroondown, aroonup = AROON(high, low, timeperiod=14)
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data['aroondown'],data['arronup'] = ta.AROON(data.high, data.low, timeperiod=timeperiod)
        return data


    def aroonosc(self,timeperiod:int=14):
        #index_38 aroonosc
        #AROONOSC - Aroon Oscillator
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型        
        data['aroonosc'] = ta.AROONOSC(data.high, data.low, timeperiod=timeperiod)
        return data

    def bop(self):
        #index_39 bop
        #BOP - Balance Of Power
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型        

        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['bop'] = ta.BOP(data.open, data.high, data.low, data.close)

        return data

    def cci(self,timeperiod:int = 14):
        #index_40 cci
        #CCI - Commodity Channel Index
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型  

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cci'] = ta.CCI(data.high,data.low, data.close, timeperiod=timeperiod)
        return  data

    def cmo(self,timeperiod:int=14):
        #index_41 cmo
        #CMO - Chande Momentum Oscillator
        #NOTE: The CMO function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cmo'] = ta.CMO(data.close, timeperiod=timeperiod)
        return  data

    def dx(self,timeperiod:int=14):
        #index_42 dx 
        #DX - Directional Movement Index
        #NOTE: The DX function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型  

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['dx'] = ta.DX(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def macd_t(self,fastperiod:int=12,slowperiod:int=26,signalperiod:int=9):
        #index_43 macd
        #MACD - Moving Average Convergence/Divergence
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['macd'],data['macdsignal'],data['macdhist'] = ta.MACD(data.close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return data

    def macdext(self,fastperiod:int=12,slowperiod:int=26,signalperiod:int=9):
        #index_44 macdext
        #MACDEXT - MACD with controllable MA type
        #macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['macd'],data['macdsignal'],data['macdhist'] = ta.MACDEXT(data.close, fastperiod=fastperiod, fastmatype=0, slowperiod=slowperiod, slowmatype=0, signalperiod=signalperiod, signalmatype=0)
        return data

    def macdfix(self,signalperiod:int=9):
        #index_45 macdfix
        #MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        #macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['macd'],data['macdsignal'],data['macdhist'] = ta.MACDFIX(data.close, signalperiod=signalperiod)
        return data

    def mfi(self,timeperiod:int=14):
        #index_46 mfi
        #MFI - Money Flow Index
        #NOTE: The MFI function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型  

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        item_variable = 'vol'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['mfi'] = ta.MFI(data.high, data.low, data.close, data.vol, timeperiod=timeperiod)
        return data


    def minus_di(self,timeperiod:int=14):
        #index_47 minus_di
        #MINUS_DI - Minus Directional Indicator
        #NOTE: The MINUS_DI function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型  

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['minus_di'] = ta.MINUS_DI(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def minus_dm(self,timeperiod:int=14):
        #index_48 minus_dm
        #MINUS_DM - Minus Directional Movement
        #NOTE: The MINUS_DM function has an unstable period.
       
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型  


        data['minus_dm'] = ta.MINUS_DM(data.high, data.low, timeperiod=timeperiod)
        return data


    def mom(self,timeperiod:int=10):
        #index_49 mom
        #MOM - Momentum
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['mom'] = ta.MOM(data.close, timeperiod=timeperiod)
        return data

    def plus_di(self,timeperiod:int=14):
        #index_50 plus_di
        #PLUS_DI - Plus Directional Indicator
        #NOTE: The PLUS_DI function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['plus_id'] = ta.PLUS_DI(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def plus_dm(self,timeperiod:int=14):
        #index_51 plus_dm
        #PLUS_DM - Plus Directional Movement
        #NOTE: The PLUS_DM function has an unstable period.
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['plus_dm'] = ta.PLUS_DM(data.high, data.low, timeperiod=timeperiod)
        return data

    def ppo(self,fastperiod:int=12,slowperiod:int=26):
        #index_52 ppo
        #PPO - Percentage Price Oscillator
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['ppo'] = ta.PPO(data.close, fastperiod=fastperiod, slowperiod=slowperiod, matype=0)
        return data

    def roc(self,timeperiod:int=10):
        #index_53 roc
        #ROC - Rate of change : ((price/prevPrice)-1)*100
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['roc'] = ta.ROC(data.close, timeperiod=timeperiod)
        return data

    def rocp(self,timeperiod:int=10):
        #index_54 rocp
        #ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        data['rocp'] = ta.ROCP(data.close, timeperiod=timeperiod)
        return data

    def rocr(self,timeperiod:int=10):
        #index_55 rocr
        #ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['rocr'] = ta.ROCR(data.close, timeperiod=timeperiod)
        return data

    def rocr100(self,timeperiod:int=10):
        #index_56 rocr100
        #ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['rocr100'] =ta.ROCR100(data.close, timeperiod=timeperiod)
        return data

    def rsi(self,timeperiod:int=14):
        #index_57 rsi
        #RSI - Relative Strength Index
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['rsi'] = ta.RSI(data.close, timeperiod=timeperiod)
        return data
    
    def stoch(self,fastk_period:int=5,slowk_period:int=3,slowd_period:int=3):
        #index_58 stoch
        #STOCH - Stochastic
        #slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['slowk'],data['slowd'] = ta.STOCH(data.high, data.low, data.close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0, slowd_period=slowd_period, slowd_matype=0)
        return data

    def stochf(self,fastk_period:int=5,fastd_period:int=3):
        #index_59 stochf
        #STOCHF - Stochastic Fast     
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['fastk'],data['fastd'] = ta.STOCHF(data.high, data.low, data.close, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=0)
        return data


    def stochrsi(self,timeperiod:int=14,fastk_period:int=5,fastd_period:int=3):
        #index_60 stochrsi
        #STOCHRSI - Stochastic Relative Strength Index  
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['fastk'],data['fastd'] = ta.STOCHRSI(data.close, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=0)
        return data

    def trix(self,timeperiod:int=30):
        #index_61 trix
        #TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['trix'] = ta.TRIX(data.close, timeperiod=timeperiod)
        return data

    def ultosc(self,timeperiod1:int=7,timeperiod2:int=14,timeperiod3:int=28):
        #index_62 stochf
        #ULTOSC - Ultimate Oscillator    
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['ultosc'] = ta.ULTOSC(data.high, data.low, data.close, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
        return data

    def willr(self,timeperiod:int=14):
        #index_63 stochf
        #WILLR - Williams' %R 
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['willr'] = ta.WILLR(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def ad(self):
        #index_64 ad
        #AD - Chaikin A/D Line

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'vol'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['ad'] = ta.AD(data.high, data.low, data.close, data.vol)
        return data

    def ad(self,fastperiod:int=3,slowperiod:int=10):
        #index_65 adosc
        #ADOSC - Chaikin A/D Oscillator

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'vol'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['adosc'] = ta.ADOSC(data.high, data.low, data.close, data.vol, fastperiod=fastperiod, slowperiod=slowperiod)
        return data

    def obv_ta(self):
        #index_66 obv
        #OBV - On Balance Volume
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'vol'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['obv'] = ta.OBV(data.close, data.vol)
        return data


    def ht_dcperiod(self):
        #index_67 ht_dcperiod
        #HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['ht_dcperiod'] = ta.HT_DCPERIOD(data.close)
        return data



    def ht_dcphase(self):
        #index_68 ht_dcphase    NNN
        #HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型


        data['ht_dcphase'] = ta.HT_DCPHASE(data.close)
        return data   

    def ht_phasor(self):
        #index_69 ht_phasor  NNNN
        #HT_PHASOR - Hilbert Transform - Phasor Components
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['inphase'],data['quadraturn'] = ta.HT_PHASOR(data.close)
        return data


    def ht_sine(self):
        #index_70 ht_sine  NNN
        #HT_SINE - Hilbert Transform - SineWave
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['sine'],data['leadsine'] = ta.HT_SINE(data.close)
        return data

    def ht_trendmode(self):
        #index_71 ht_trendmode
        #HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['ht_trendmode'] = ta.HT_TRENDMODE(data.close) 
        return data

    def avgprice(self):
        #index_72 AVGPRICE
        #AVGPRICE - Average Price

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['avgprice'] = ta.AVGPRICE(data.open, data.high, data.low, data.close)
        return data

    def medprice(self):
        #index_73 MEDPRICE
        #MEDPRICE - Median Price

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['medprice'] = ta.MEDPRICE(data.high, data.low)
        return data

    def typprice(self):
        #index_74 TYPPRICE
        #TYPPRICE - Typical Price

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['typprice'] = ta.TYPPRICE(data.high, data.low, data.close)
        return data

    def wclprice(self):
        #index_75 WCLPRICE
        #WCLPRICE - Weighted Close Price

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['wclprice'] = ta.WCLPRICE(data.high, data.low, data.close)
        return data

    def atr(self,timeperiod:int=14):
        #index_76 atr
        #ATR - Average True Range

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['atr'] = ta.ATR(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def natr(self,timeperiod:int=14):
        #index_77 aatr
        #NATR - Normalized Average True Range

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['atr'] = ta.NATR(data.high, data.low, data.close, timeperiod=timeperiod)
        return data

    def trange(self):
        #index_78 trange
        #TRANGE - True Range

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['trange'] = ta.TRANGE(data.high, data.low, data.close)
        return data

    def cdl2crows(self):
        #index_79 cdl2crows      
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdl2crows'] = ta.CDL2CROWS(data.open, data.high, data.low, data.close)
        return data

    def cdl3blackcrows(self):
        #index_80 cdl3blackcrows
        #CDL3BLACKCROWS - Three Black Crows
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdl3blackcrows'] = ta.CDL3BLACKCROWS(data.open, data.high, data.low, data.close)
        return data

    def cdl3inside(self):
        #index_81 cdl3blackcrows
        #CDL3INSIDE - Three Inside Up/Down
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdl3inside'] = ta.CDL3INSIDE(data.open, data.high, data.low, data.close)
        return data

    def cdl3linestrike(self):
        #index_82 cdl3cdl3linestrike
        #CDL3INSIDE - Three Inside Up/Down
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdl3linestrike'] = ta.CDL3LINESTRIKE(data.open, data.high,data.low, data.close)
        return data

    def cdl3outside(self):
        #index_83 cdl3outside
        #CDL3OUTSIDE - Three Outside Up/Down
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdl3outside'] = ta.CDL3OUTSIDE(data.open, data.high,data.low, data.close)
        return data


    def cdl3starstinsouth(self):
        #index_84 CDL3STARSINSOUTH
        #CDL3STARSINSOUTH - Three Stars In The South
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdl3starsinsouth'] = ta.CDL3STARSINSOUTH(data.open, data.high,data.low, data.close)
        return data

    def cdl3whitesoldiers(self):
        #index_85  CDL3WHITESOLDIERS
        #CDL3WHITESOLDIERS - Three Advancing White Soldiers
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdl3whitesoldiersh'] = ta.CDL3WHITESOLDIERS(data.open, data.high,data.low, data.close)
        return data

    def cdlabandonedbaby(self):
        #index_86  CDLABANDONEDBABY
        #CDLABANDONEDBABY - Abandoned Baby
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlabandonedbaby'] = ta.CDLABANDONEDBABY(data.open, data.high,data.low, data.close,penetration=0)
        return data


    def cdladvanceblock(self):
        #index_87 CDLADVANCEBLOCK
        #CDLADVANCEBLOCK - Advance Block
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdladanceblock'] = ta.CDLADVANCEBLOCK(data.open, data.high,data.low, data.close)
        return data

    def cdlbelthold(self):
        #index_88 CDLBELTHOLD
        #CDLBELTHOLD - Belt-hold
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlbelthold'] = ta.CDLBELTHOLD(data.open, data.high,data.low, data.close)
        return data

    def cdlbreakaway(self):
        #index_89 CDLBREAKAWAY
        #CDLBREAKAWAY - Breakaway
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlbreakaway'] = ta.CDLBREAKAWAY(data.open, data.high,data.low, data.close)
        return data


    def cdlclosingmarubozu(self):
        #index_90 CDLCLOSINGMARUBOZU
        #CDLCLOSINGMARUBOZU - Closing Marubozu
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlclostingmarubozu'] = ta.CDLCLOSINGMARUBOZU(data.open, data.high,data.low, data.close)
        return data


    def cdlconcealbabyswall(self):
        #index_91 CDLCLOSINGMARUBOZU
        #CDLCONCEALBABYSWALL - Concealing Baby Swallow
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlconcealbabyswall'] = ta.CDLCONCEALBABYSWALL(data.open, data.high,data.low, data.close)
        return data


    def cdlcounterattack(self):
        #index_92 CDLCOUNTERATTACK
        #CDLCOUNTERATTACK - Counterattack
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlcounterattack'] = ta.CDLCOUNTERATTACK(data.open, data.high,data.low, data.close)
        return data

    def cdldarkcloundcover(self):
        #index_93 CDLDARKCLOUDCOVER
        #CDLDARKCLOUDCOVER - Dark Cloud Cover

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlcounterattack'] = ta.CDLDARKCLOUDCOVER(data.open, data.high,data.low, data.close)
        return data

    def cdldoji(self):
        #index_94 CDLDOJI 
        #CDLDOJI - Doji

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdldoji'] = ta.CDLDOJI(data.open, data.high,data.low, data.close)
        return data
    
    def cdldojistar(self):
        #index_95 CDLDOJI 
        #CDLDOJISTAR - Doji Star

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdldojistar'] = ta.CDLDOJISTAR(data.open, data.high,data.low, data.close)
        return data


    def cdldragonflydoji(self):
        #index_96 CDLDRAGONFLYDOJI
        #CDLDRAGONFLYDOJI - Dragonfly Doji

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdldragonflydoji'] = ta.CDLDRAGONFLYDOJI(data.open, data.high,data.low, data.close)
        return data

    def cdlengulfing(self):
        #index_97 CDLDRAGONFLYDOJI
        #CDLENGULFING - Engulfing Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlengulfing'] = ta.CDLENGULFING(data.open, data.high,data.low, data.close)
        return data

    def cdleveningdojistar(self):
        #index_98 CDLEVENINGDOJISTAR
        #CDLEVENINGDOJISTAR - Evening Doji Star

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdleveningdojistar'] = ta.CDLEVENINGDOJISTAR(data.open, data.high,data.low, data.close,penetration=0)
        return data

    def cdleveningstar(self):
        #index_99 CDLEVENINGSTAR 
        #CDLEVENINGSTAR - Evening Star

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdleveningstar'] = ta.CDLEVENINGSTAR(data.open, data.high,data.low, data.close,penetration=0)
        return data

    def cdlgapsidesidewhite(self):
        #index_100 CDLGAPSIDESIDEWHITE 
        #CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlgapsidesidewhite'] = ta.CDLGAPSIDESIDEWHITE(data.open, data.high,data.low, data.close)
        return data


    def cdlgravestonedoji(self):
        #index_101 CDLGRAVESTONEDOJI - Gravestone Doji
        #CDLGRAVESTONEDOJI - Gravestone Doji

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(data.open, data.high,data.low, data.close)
        return data

    def cdlhammer(self):
        #index_102 CDLHAMMER  
        #CDLHAMMER - Hammer

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlhammer'] = ta.CDLHAMMER(data.open, data.high,data.low, data.close)
        return data

    def cdlhangingman(self):
        #index_103 CDLHANGINGMAN  
        #CDLHANGINGMAN - Hanging Man

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlhangingman'] = ta.CDLHANGINGMAN(data.open, data.high,data.low, data.close)
        return data


    def cdlharami(self):
        #index_104 CDLHARAMI 
        #CDLHARAMI - Harami Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlharami'] = ta.CDLHARAMI(data.open, data.high,data.low, data.close)
        return data


    def cdlharamicross(self):
        #index_105 CDLHARAMICROSS 
        #CDLHARAMICROSS - Harami Cross Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlharamicross'] = ta.CDLHARAMICROSS(data.open, data.high,data.low, data.close)
        return data

    def cdlhighwave(self):
        #index_106 CDLHIGHWAVE 
        #CDLHIGHWAVE - High-Wave Candle

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlhighwave'] = ta.CDLHIGHWAVE(data.open, data.high,data.low, data.close)
        return data



    def cdlhikkake(self):
        #index_107 CDLHIKKAKE
        #CDLHIKKAKE - Hikkake Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlhikkake'] = ta.CDLHIKKAKE(data.open, data.high,data.low, data.close)
        return data


    def cdlhikkakemod(self):
        #index_108 CDLHIKKAKEMOD 
        #CDLHIKKAKEMOD - Modified Hikkake Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlhikkakemod'] = ta.CDLHIKKAKEMOD(data.open, data.high,data.low, data.close)
        return data

    def cdlhomingpigeon(self):
        #index_109 CDLHOMINGPIGEON  
        #CDLHOMINGPIGEON - Homing Pigeon

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlhomingpigeon'] = ta.CDLHOMINGPIGEON(data.open, data.high,data.low, data.close)
        return data

    def cdlidentical3crows(self):
        #index_110 CDLIDENTICAL3CROWS  
        #CDLIDENTICAL3CROWS - Identical Three Crows

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlidentical3crows'] = ta.CDLIDENTICAL3CROWS(data.open, data.high,data.low, data.close)
        return data

    def cdlinneck(self):
        #index_111 CDLINNECK 
        #CDLINNECK - In-Neck Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlinneck'] = ta.CDLINNECK(data.open, data.high,data.low, data.close)
        return data

    def cdlinvertedhammer(self):
        #index_112 CDLINVERTEDHAMMER  
        #CDLINVERTEDHAMMER - Inverted Hammer

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(data.open, data.high,data.low, data.close)
        return data

    def cdlkicking(self):
        #index_113 CDLKICKING - Kicking
        #CDLKICKING - Kicking

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlkickingr'] = ta.CDLKICKING(data.open, data.high,data.low, data.close)
        return data


    def cdlkickingbylegngth(self):
        #index_114 CDLKICKINGBYLENGTH   
        #CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlkickingbylength'] = ta.CDLKICKINGBYLENGTH(data.open, data.high,data.low, data.close)
        return data

    def cdlladderbottom(self):
        #index_115 CDLLADDERBOTTOM  
        #CDLLADDERBOTTOM - Ladder Bottom

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlladderbottom'] = ta.CDLLADDERBOTTOM(data.open, data.high,data.low, data.close)
        return data


    def cdllongleggeddoji(self):
        #index_116 CDLLONGLEGGEDDOJI  
        #CDLLONGLEGGEDDOJI - Long Legged Doji

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdllongleggeddoji'] = ta.CDLLONGLEGGEDDOJI(data.open, data.high,data.low, data.close)
        return data

    def cdllongline(self):
        #index_117 CDLLONGLINE  
        #CDLLONGLINE - Long Line Candle

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdllongline'] = ta.CDLLONGLINE(data.open, data.high,data.low, data.close)
        return data

    def cdlmarubozu(self):
        #index_118 CDLMARUBOZU  
        #CDLMARUBOZU - Marubozu

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlmarubozu'] = ta.CDLMARUBOZU(data.open, data.high,data.low, data.close)
        return data

    def cdlmatchinglow(self):
        #index_119 CDLMATCHINGLOW - Matching Low
        #CDLMARUBOZU - Marubozu

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlmatchinglow'] = ta.CDLMATCHINGLOW(data.open, data.high,data.low, data.close)
        return data

    def cdlmathold(self):
        #index_119 CDLMATCHINGLOW - Matching Low
        #CDLMATHOLD - Mat Hold

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlmathold'] = ta.CDLMATHOLD(data.open, data.high,data.low, data.close,penetration=0)
        return data


    def cdlmorningdojistar(self):
        #index_120 CDLMORNINGDOJISTAR 
        #CDLMORNINGDOJISTAR - Morning Doji Star

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlmorningdojistar'] = ta.CDLMORNINGDOJISTAR(data.open, data.high,data.low, data.close,penetration=0)
        return data

    def cdlmorningstar(self):
        #index_121 CDLMORNINGSTAR 
        #CDLMORNINGSTAR - Morning Star

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlomrningstar'] = ta.CDLMORNINGSTAR(data.open, data.high,data.low, data.close,penetration=0)
        return data

    def cdlonneck(self):
        #index_122 CDLONNECK - On-Neck Pattern
        #CDLONNECK - On-Neck Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlonneck'] = ta.CDLONNECK(data.open, data.high,data.low, data.close)
        return data

    def cdlpiercting(self):
        #index_123 CDLPIERCING - Piercing Pattern
        #CDLPIERCING - Piercing Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlpiercing'] = ta.CDLPIERCING(data.open, data.high,data.low, data.close)
        return data

    def cdlrickshawman(self):
        #index_124 CDLRICKSHAWMAN 
        #CDLRICKSHAWMAN - Rickshaw Man

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlrickshawman'] = ta.CDLRICKSHAWMAN(data.open, data.high,data.low, data.close)
        return data

    def cdlrisefall3methods(self):
        #index_125 CDLRISEFALL3METHODS  
        #CDLRISEFALL3METHODS - Rising/Falling Three Methods

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlrisefall3methods'] = ta.CDLRISEFALL3METHODS(data.open, data.high,data.low, data.close)
        return data


    def cdlseparatinglines(self):
        #index_126 CDLSEPARATINGLINES  
        #CDLSEPARATINGLINES - Separating Lines

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlseparatinglines'] = ta.CDLSEPARATINGLINES(data.open, data.high,data.low, data.close)
        return data

    def cdlshootingstar(self):
        #index_127 CDLSHOOTINGSTAR  
        #CDLSHOOTINGSTAR - Shooting Star

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(data.open, data.high,data.low, data.close)
        return data

    def cdlshortline(self):
        #index_128 CDLSHORTLINE - Short Line Candle
        #CDLSHORTLINE - Short Line Candle

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlshortline'] = ta.CDLSHORTLINE(data.open, data.high,data.low, data.close)
        return data

    def cdlspinningtop(self):
        #index_129 CDLSPINNINGTOP - Spinning Top
        #CDLSPINNINGTOP - Spinning Top

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlspinningtop'] = ta.CDLSPINNINGTOP(data.open, data.high,data.low, data.close)
        return data

    def cdlstalledpattern(self):
        #index_130 CDLSTALLEDPATTERN - Stalled Pattern
        #CDLSTALLEDPATTERN - Stalled Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlstalledpattern'] = ta.CDLSTALLEDPATTERN(data.open, data.high,data.low, data.close)
        return data

    def cdlsticksandwich(self):
        #index_131 CDLSTICKSANDWICH  
        #CDLSTICKSANDWICH - Stick Sandwich

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlsticksandwich'] = ta.CDLSTICKSANDWICH(data.open, data.high,data.low, data.close)
        return data

    def cdltakuri(self):
        #index_132 CDLTAKURI
        #CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdltakuri'] = ta.CDLTAKURI(data.open, data.high,data.low, data.close)
        return data

    def cdltasukigap(self):
        #index_132 CDLTASUKIGAP  
        #CDLTASUKIGAP - Tasuki Gap

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdltasukigap'] = ta.CDLTASUKIGAP(data.open, data.high,data.low, data.close)
        return data

    def cdlthrusting(self):
        #index_133 CDLTHRUSTING  
        #CDLTHRUSTING - Thrusting Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlthrustingp'] = ta.CDLTHRUSTING(data.open, data.high,data.low, data.close)
        return data

    def cdltristar(self):
        #index_134 CDLTRISTAR  
        #CDLTRISTAR - Tristar Pattern

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdltristar'] = ta.CDLTRISTAR(data.open, data.high,data.low, data.close)
        return data

    def cdlunique3river(self):
        #index_135 CDLUNIQUE3RIVER  
        #CDLUNIQUE3RIVER - Unique 3 River

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlunique3river'] = ta.CDLUNIQUE3RIVER(data.open, data.high,data.low, data.close)
        return data

    def cdlupsidegap2crows(self):
        #index_136 CDLUPSIDEGAP2CROWS  
        #CDLUPSIDEGAP2CROWS - Upside Gap Two Crows

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlupsidegap2crows'] = ta.CDLUPSIDEGAP2CROWS(data.open, data.high,data.low, data.close)
        return data

    def cdlxsidegap3methods(self):
        #index_137 CDLXSIDEGAP3METHODS  
        #CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排
        
        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'open'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['cdlxsidegap3methods'] = ta.CDLXSIDEGAP3METHODS(data.open, data.high,data.low, data.close)
        return data


    def beta(self,timeperiod:int=5):
        #index_138 BETA 
        #BETA - Beta

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['beta'] = ta.BETA(data.high, data.low, timeperiod=timeperiod)
        return data

    def correl(self,timeperiod:int=30):
        #index_139 CORREL            
        #CORREL - Pearson's Correlation Coefficient (r)
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型

        data['correl'] = ta.CORREL(data.high,data.low,timeperiod=timeperiod)
        return data

    def linearreg(self,timeperiod:int=14):
        #index_140 LINEARREG
        #LINEARREG - Linear Regression

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['linearreg'] = ta.LINEARREG(data.close, timeperiod=timeperiod)

        return data

    def linearreg_angle(self,timeperiod:int=14):
        #index_141 LINEARREG_ANGLE
        #LINEARREG_ANGLE - Linear Regression Angle

        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['linearreg_angle'] = ta.LINEARREG_ANGLE(data.close, timeperiod=timeperiod)

        return data

    def linearreg_intercept(self,timeperiod:int=14):
        #index_142 LINEARREG_INTERCEPT 
        #LINEARREG_INTERCEPT - Linear Regression Intercept
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['linearreg_intercept'] = ta.LINEARREG_INTERCEPT(data.close, timeperiod=timeperiod)

        return data    

    def linearreg_slope(self,timeperiod:int=14):
        #index_143 LINEARREG_SLOPE 
        #LINEARREG_SLOPE - Linear Regression Slope
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['linearreg_slope'] = ta.LINEARREG_SLOPE(data.close, timeperiod=timeperiod)

        return data 

    def stddev(self,timeperiod:int=5,nbdev:int=1):
        #index_144 STDDEV
        #STDDEV - Standard Deviation
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['stddev'] = ta.STDDEV(data.close, timeperiod=timeperiod,nbdev=nbdev)

        return data 


    def tsf(self,timeperiod:int=14):
        #index_145 TSF 
        #TSF - Time Series Forecast
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['tsf'] = ta.TSF(data.close, timeperiod=timeperiod)

        return data 

    def var_ta(self,timeperiod:int=5,nbdev:int=1):
        #index_146 VAR 
        #VAR - Variance
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['var'] = ta.VAR(data.close, timeperiod=timeperiod, nbdev=nbdev)

        return data



    def acos(self):
        #index_147 ACOS 
        #ACOS - Vector Trigonometric ACos
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['acos'] = ta.ACOS(data.close)

        return data

    def asin(self):
        #index_148 ASIN 
        #ASIN - Vector Trigonometric ASin
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['asin'] = ta.ASIN(data.close)

        return data

    def atan(self):
        #index_149 ATAN 
        #ATAN - Vector Trigonometric ATan
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['atan'] = ta.ATAN(data.close)

        return data

    def ceil(self):
        #index_150 CEIL
        #CEIL - Vector Ceil
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['ceil'] = ta.CEIL(data.close)

        return data

    def cos(self):
        #index_151 COS
        #COS - Vector Trigonometric Cos
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['cos'] = ta.COS(data.close)

        return data


    def cosh(self):
        #index_152 COSH 
        #COSH - Vector Trigonometric Cosh
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['cosh'] = ta.COSH(data.close)

        return data


    def exp(self):
        #index_152 EXP
        #EXP
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['exp'] = ta.EXP(data.close)

        return data

    def floor(self):
        #index_153 FLOOR 
        #FLOOR - Vector Floor
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['floor'] = ta.FLOOR(data.close)

        return data

    def ln(self):
        #index_154 LN 
        #LN - Vector Log Natural
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['ln'] = ta.LN(data.close)

        return data


    def log10(self):
        #index_155 LOG10 
        #LOG10 - Vector Log10
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['log10'] = ta.LOG10(data.close)

        return data


    def sin(self):
        #index_156 SIN
        #SIN - Vector Trigonometric Sin
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['sin'] = ta.SIN(data.close)

        return data


    def sinh(self):
        #index_157 SINH 
        #SINH - Vector Trigonometric Sinh
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['sinh'] = ta.SINH(data.close)

        return data

    def sqrt(self):
        #index_158 SQRT 
        #SQRT - Vector Square Root
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['sqrt'] = ta.SQRT(data.close)

        return data


    def tan(self):
        #index_159 TAN 
        #TAN - Vector Trigonometric Tan
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['tan'] = ta.TAN(data.close)

        return data

    def tanh(self):
        #index_160 TANH 
        #TANH - Vector Trigonometric Tanh
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        
        data['tanh'] = ta.TANH(data.close)

        return data

    def add(self):
        #index_161  ADD 
        #ADD - Vector Arithmetic Add
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型        

        data['add'] = ta.ADD(data.high,data.low)

        return data


    def div(self):
        #index_162 DIV 
        #DIV - Vector Arithmetic Div
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型        

        data['div'] = ta.DIV(data.high,data.low)

        return data


    def max(self,timeperiod:int=30):
        #index_163 MAX 
        #MAX - Highest value over a specified period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型       

        data['max'] = ta.MAX(data.close,timeperiod=timeperiod)

        return data


    def maxindex(self,timeperiod:int=30):
        #index_164 MAXINDEX 
        #MAXINDEX - Index of highest value over a specified period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型       

        data['maxindex'] = ta.MAXINDEX(data.close,timeperiod=timeperiod)

        return data


    def min(self,timeperiod:int=30):
        #index_165 MIN 
        #MIN - Lowest value over a specified period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型       

        data['min'] = ta.MIN(data.close,timeperiod=timeperiod)

        return data


    def minindex(self,timeperiod:int=30):
        #index_166 MININDEX 
        #MININDEX - Index of lowest value over a specified period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型       

        data['minindex'] = ta.MININDEX(data.close,timeperiod=timeperiod)

        return data

    def minmax(self,timeperiod:int=30):
        #index_167 MINMAX 
        #MINMAX - Lowest and highest values over a specified period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型       

        data['min'],data['max'] = ta.MINMAX(data.close,timeperiod=timeperiod)

        return data


    def minmaxindex(self,timeperiod:int=30):
        #index_168 MINMAXINDEX 
        #MINMAXINDEX - Indexes of lowest and highest values over a specified period
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型       

        data['minindex'],data['maxindex'] = ta.MINMAXINDEX(data.close,timeperiod=timeperiod)

        return data

    def mult(self):
        #index_169 MULT - Vector Arithmetic Mult
        #MULT - Vector Arithmetic Mult
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型          

        data['mult'] = ta.MULT(data.high,data.low)

        return data


    def sub(self):
        #index_170 SUB 
        #SUB - Vector Arithmetic Substraction
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'high'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型
        item_variable = 'low'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型          

        data['sub'] = ta.SUB(data.high,data.low)

        return data


    def sum(self,timeperiod:int=30):
        #index_171 SUM 
        #SUM - Summation
        data = self.df_wm[(self.df_wm['ts_code']==self.ts_code)&(self.df_wm['trade_date']>=self.start_dt)&(self.df_wm['trade_date']<=self.given_dt)] 
        data.drop_duplicates(subset=['ts_code','trade_date'],keep='first',inplace=True)  #去重
        data = data.reset_index()   #重新排        

        item_variable = 'close'   #设定转换类型的变更
        data[item_variable] = data[item_variable].astype('float')  #转换数据类型        

        data['sum'] = ta.SUM(data.close,timeperiod=timeperiod)

        return data




    
#if __name__ == '__main__':
    
#a = WmDb('20200101','20200308')



