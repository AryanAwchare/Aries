//+------------------------------------------------------------------+
//| ea_bridge.mq5                                                    |
//| Aries gold platform - local MetaTrader execution bridge          |
//|                                                                  |
//| Polls the platform's FastAPI server for orders and reports fills.|
//| Protocol (plain text, no JSON parsing needed):                   |
//|   GET  <host>:<port>/ea/next-order [?key=...]                    |
//|        -> "ORDER|<external_id>|<symbol>|<side>|<volume>|<sl>|<tp>"|
//|        -> "NONE" | "AUTHFAIL"                                    |
//|   POST <host>:<port>/ea/execution [?key=...]                     |
//|        body: "RESULT|<external_id>|1|<broker_order_id>|<fill>"  |
//|              "RESULT|<external_id>|0|||<error message>"          |
//|                                                                  |
//| Setup: MetaTrader 5 -> Tools -> Options -> Expert Advisors ->    |
//|   "Allow WebRequest for listed URL:" add http://127.0.0.1:8000   |
//+------------------------------------------------------------------+
#property copyright "Aries gold platform"
#property version   "1.00"

#include <Trade\Trade.mqh>

input string InpHost    = "http://127.0.0.1";  // Platform host
input int    InpPort    = 8000;               // Platform port
input string InpApiKey  = "";                 // Platform EA_API_KEY (optional)
input ulong  InpSlipPts = 50;                 // Slippage (points)
input int    InpTimeout = 3000;               // HTTP timeout (ms)
input bool   InpLog     = true;               // Print activity to the journal

CTrade g_trade;
string g_url_next = "";
string g_url_exec = "";
bool   g_busy      = false;
datetime g_last_poll = 0;

//+------------------------------------------------------------------+
string HttpGet(const string url)
  {
   uchar data[], result[];
   string headers;
   ResetLastError();
   int code = WebRequest("GET", url, "", "", InpTimeout, data, 0, result, headers);
   if(code == -1)
     {
      if(InpLog)
         Print("[EA] GET error ", GetLastError());
      return("");
     }
   return(CharArrayToString(result));
  }

//+------------------------------------------------------------------+
string HttpPost(const string url, const string body)
  {
   uchar data[], result[];
   StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   int size = ArraySize(data) - 1; // drop the trailing null terminator
   string headers;
   ResetLastError();
   int code = WebRequest("POST", url, "", "", InpTimeout, data, size, result, headers);
   if(code == -1)
     {
      if(InpLog)
         Print("[EA] POST error ", GetLastError());
      return("");
     }
   return(CharArrayToString(result));
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   string base = InpHost + ":" + IntegerToString(InpPort);
   g_url_next = base + "/ea/next-order";
   g_url_exec = base + "/ea/execution";
   if(InpApiKey != "")
     {
      g_url_next += "?key=" + InpApiKey;
      g_url_exec += "?key=" + InpApiKey;
     }
   g_trade.SetDeviationInPoints(InpSlipPts);
   Print("[EA] bridge up. next=", g_url_next);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   if(g_busy)
      return;
   // rate-limit ourselves to roughly one request per second
   if(TimeCurrent() == g_last_poll)
      return;
   g_last_poll = TimeCurrent();
   g_busy = true;
   string result = HttpGet(g_url_next);
   g_busy = false;
   if(result == "")
      return;
   if(result == "NONE" || result == "AUTHFAIL")
      return;
   if(StringFind(result, "ORDER|") == 0)
      HandleOrder(result);
  }

//+------------------------------------------------------------------+
void HandleOrder(const string payload)
  {
   string parts[];
   int n = StringSplit(payload, '|', parts);
   if(n < 7)
     {
      Print("[EA] malformed order payload: ", payload);
      return;
     }
   string ext    = parts[1];
   string symbol = parts[2];
   string side   = parts[3];
   double volume = StringToDouble(parts[4]);
   double sl     = ParsePrice(parts[5]);
   double tp     = ParsePrice(parts[6]);
   Print("[EA] execute ", side, " ", symbol, " ", DoubleToString(volume, 2),
         " sl=", DoubleToString(sl, _Digits), " tp=", DoubleToString(tp, _Digits));
   Execute(ext, symbol, side, volume, sl, tp);
  }

//+------------------------------------------------------------------+
void Execute(string ext, string symbol, string side, double volume,
             double sl, double tp)
  {
   if(!SymbolSelect(symbol, true))
     {
      Report(ext, false, "", 0.0, "symbol not available in Market Watch: " + symbol);
      return;
     }
   double price = (side == "buy")
                  ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(symbol, SYMBOL_BID);
   if(price <= 0.0)
     {
      Report(ext, false, "", 0.0, "no price for " + symbol);
      return;
     }
   string comment = "aries-" + ext;
   bool ok;
   if(side == "buy")
      ok = g_trade.Buy(volume, symbol, price, sl, tp, comment);
   else if(side == "sell")
      ok = g_trade.Sell(volume, symbol, price, sl, tp, comment);
   else
     {
      Report(ext, false, "", 0.0, "unknown side: " + side);
      return;
     }
   if(ok)
     {
      ulong ticket = g_trade.ResultOrder();
      double fill  = g_trade.ResultPrice();
      Report(ext, true, IntegerToString(ticket), fill, "");
     }
   else
      Report(ext, false, "", 0.0,
             StringFormat("OrderSend retcode=%d %s", g_trade.ResultRetcode(),
                          g_trade.ResultRetcodeDescription()));
  }

//+------------------------------------------------------------------+
void Report(string ext, bool ok, string ticket, double fill, string error)
  {
   string body = ok
                 ? StringFormat("RESULT|%s|1|%s|%.6g", ext, ticket, fill)
                 : StringFormat("RESULT|%s|0|||%s", ext, error);
   string resp = HttpPost(g_url_exec, body);
   if(InpLog)
      Print("[EA] report ", ok ? "filled":"rejected", " ", ext, " -> ", resp);
  }

//+------------------------------------------------------------------+
double ParsePrice(const string value)
  {
   if(value == "")
      return(0.0);
   double v = StringToDouble(value);
   return(v);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Print("[EA] bridge stopped (", reason, ")");
  }
//+------------------------------------------------------------------+