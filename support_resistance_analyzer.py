import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf
import scipy
import scipy.signal
import scipy.stats
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def find_levels( 
        price: np.array, atr: float, # Log closing price, and log atr 
        first_w: float = 0.1, 
        atr_mult: float = 3.0, 
        prom_thresh: float = 0.1,
        min_data_points: int = 10
):
    """Trouve les niveaux de support/résistance basés sur la densité des prix"""
    
    # Validation des données d'entrée
    if len(price) < min_data_points:
        print(f"Warning: Pas assez de données ({len(price)} < {min_data_points})")
        return [], [], {}, [], [], []
    
    if atr <= 0 or not np.isfinite(atr):
        # Utiliser la volatilité empirique comme fallback
        atr = np.std(np.diff(price)) * 2.0
        if atr <= 0:
            atr = 0.01  # Valeur minimale de sécurité
    
    # Setup weights - plus de poids aux prix récents
    last_w = 1.0
    w_step = (last_w - first_w) / len(price)
    weights = first_w + np.arange(len(price)) * w_step
    weights[weights < 0] = 0.0
    
    # Normaliser les poids
    weights = weights / np.sum(weights)

    try:
        # Get kernel of price (estimation de densité)
        # Ajuster la bande passante pour les petites fenêtres
        bw = max(atr * atr_mult, np.std(price) * 0.5)
        kernel = scipy.stats.gaussian_kde(price, bw_method=bw, weights=weights)

        # Construct market profile
        min_v = np.min(price)
        max_v = np.max(price)
        
        # Ajuster le nombre de points selon la taille des données
        n_points = min(200, max(50, len(price) * 2))
        step = (max_v - min_v) / n_points
        
        if step <= 0:
            print("Warning: Plage de prix trop petite")
            return [], [], {}, [], [], weights
            
        price_range = np.arange(min_v, max_v, step)
        pdf = kernel(price_range) # Market profile

        # Find significant peaks in the market profile
        pdf_max = np.max(pdf)
        
        # Ajuster le seuil de prominence pour les petites fenêtres
        prom_min = pdf_max * max(prom_thresh, 0.05)
        
        # Paramètres adaptatifs pour find_peaks
        min_distance = max(1, len(price_range) // 20)
        
        peaks, props = scipy.signal.find_peaks(
            pdf, 
            prominence=prom_min,
            distance=min_distance,
            height=pdf_max * 0.1
        )
        
        levels = [] 
        for peak in peaks:
            if 0 <= peak < len(price_range):
                levels.append(np.exp(price_range[peak]))

        return levels, peaks, props, price_range, pdf, weights
        
    except Exception as e:
        print(f"Erreur dans find_levels: {e}")
        return [], [], {}, [], [], weights


def find_pivot_points(high: np.array, low: np.array, window: int = 5):
    """Trouve les points de pivot (hauts et bas) dans les données"""
    
    # Ajuster la fenêtre selon la taille des données
    if len(high) < window * 3:
        window = max(1, len(high) // 5)
    
    highs = []
    lows = []
    
    if window < 1:
        return highs, lows
    
    for i in range(window, len(high) - window):
        # Pivot high: le prix est plus haut que les 'window' valeurs de chaque côté
        local_high = high[i-window:i+window+1]
        if len(local_high) > 0 and high[i] == np.max(local_high):
            highs.append((i, high[i]))
        
        # Pivot low: le prix est plus bas que les 'window' valeurs de chaque côté
        local_low = low[i-window:i+window+1]
        if len(local_low) > 0 and low[i] == np.min(local_low):
            lows.append((i, low[i]))
    
    return highs, lows


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array, tolerance: float = 1e-5):
    """Vérifie la validité d'une ligne de tendance avec tolérance adaptative"""
    
    if len(y) == 0 or pivot >= len(y) or pivot < 0:
        return -1.0
    
    # Ajuster la tolérance selon la volatilité des données
    volatility = np.std(y)
    adaptive_tolerance = max(tolerance, volatility * 0.01)
    
    # Trouve l'ordonnée à l'origine de la ligne passant par le pivot avec la pente donnée
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Vérifie si la ligne est valide avec tolérance adaptative
    if support and diffs.max() > adaptive_tolerance:
        return -1.0
    elif not support and diffs.min() < -adaptive_tolerance:
        return -1.0

    # Somme des carrés des différences
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    """Optimise la pente d'une ligne de tendance avec gestion améliorée"""
    
    if len(y) < 3 or pivot >= len(y) or pivot < 0:
        return None, None
    
    # Ajuster l'unité de pente selon la taille des données
    slope_unit = (y.max() - y.min()) / max(len(y), 10)
    
    # Variables d'optimisation adaptatives
    opt_step = 1.0
    min_step = 0.001  # Augmenté pour éviter les boucles infinies
    curr_step = opt_step
    max_iterations = 50  # Limite d'itérations
    
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    if best_err < 0:
        return None, None  # Pente initiale invalide

    get_derivative = True
    derivative = None
    iterations = 0
    
    while curr_step > min_step and iterations < max_iterations:
        iterations += 1

        if get_derivative:
            # Différentiation numérique
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                break  # Impossible d'optimiser

            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            curr_step *= 0.5
        else:
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True
    
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    """Ajuste les lignes de tendance de support et résistance avec gestion d'erreurs"""
    
    if len(high) < 3 or len(low) < 3 or len(close) < 3:
        return (None, None)
    
    try:
        x = np.arange(len(close))
        
        # Vérifier que les données sont valides
        if np.any(~np.isfinite(close)) or np.any(~np.isfinite(high)) or np.any(~np.isfinite(low)):
            return (None, None)
        
        coefs = np.polyfit(x, close, 1)
        line_points = coefs[0] * x + coefs[1]
        
        # Trouver les pivots avec gestion d'erreurs
        high_diffs = high - line_points
        low_diffs = low - line_points
        
        if len(high_diffs) == 0 or len(low_diffs) == 0:
            return (None, None)
            
        upper_pivot = np.argmax(high_diffs)
        lower_pivot = np.argmin(low_diffs)
        
        support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
        resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

        return (support_coefs, resist_coefs)
        
    except Exception as e:
        print(f"Erreur dans fit_trendlines_high_low: {e}")
        return (None, None)


def calculate_sr_levels(data: pd.DataFrame, lookback: int, 
                       first_w: float = 0.01, atr_mult: float = 3.0, 
                       prom_thresh: float = 0.25):
    """Calcule les niveaux de support/résistance pour chaque période avec gestion améliorée"""
    
    # Validation des paramètres d'entrée
    if lookback < 5:
        print(f"Warning: Lookback très petit ({lookback}), ajusté à 5")
        lookback = 5
    
    if len(data) < lookback + 1:
        raise ValueError(f"Pas assez de données ({len(data)}) pour lookback {lookback}")
    
    # Vérifier que les données sont valides
    if (data <= 0).any().any():
        print("Warning: Suppression des valeurs nulles ou négatives")
        data = data[(data > 0).all(axis=1)]
        if len(data) < lookback + 1:
            raise ValueError("Pas assez de données valides après nettoyage")
    
    # Calcul de l'ATR en log avec gestion d'erreurs améliorée
    try:
        # Convertir en log avec vérification
        log_high = np.log(data['high'])
        log_low = np.log(data['low'])
        log_close = np.log(data['close'])
        
        # Vérifier les valeurs infinies ou NaN
        if np.any(~np.isfinite(log_high)) or np.any(~np.isfinite(log_low)) or np.any(~np.isfinite(log_close)):
            raise ValueError("Valeurs infinies détectées dans les logs")
            
        atr = ta.atr(log_high, log_low, log_close, lookback)
        
        # Remplacer les NaN par une estimation de volatilité
        if atr.isna().any():
            returns = log_close.diff()
            vol_estimate = returns.rolling(window=min(lookback, 10), min_periods=2).std()
            atr = atr.fillna(vol_estimate * np.sqrt(lookback))
            
    except Exception as e:
        print(f"Erreur dans le calcul ATR: {e}")
        # ATR de fallback basé sur la volatilité simple
        returns = np.log(data['close']).diff()
        atr = returns.rolling(window=min(lookback, 10), min_periods=2).std() * np.sqrt(lookback)
        atr = atr.fillna(method='bfill').fillna(0.01)
    
    # Calcul des lignes de tendance avec gestion d'erreurs
    data_log = np.log(data[['high', 'low', 'close', 'open']])
    
    # Initialisation des résultats
    n_periods = len(data)
    support_slopes = [np.nan] * n_periods
    resist_slopes = [np.nan] * n_periods
    support_intercepts = [np.nan] * n_periods
    resist_intercepts = [np.nan] * n_periods
    density_levels = [None] * n_periods
    
    # Points de pivot
    pivot_highs_all = []
    pivot_lows_all = []
    
    # Calcul avec pas adaptatif pour les petites fenêtres
    step_size = max(1, lookback // 10) if lookback < 50 else max(5, lookback // 20)
    
    for i in range(lookback, n_periods, step_size):
        # Remplir les indices intermédiaires si nécessaire
        if step_size > 1:
            fill_start = max(lookback, i - step_size + 1)
            fill_end = i + 1
        else:
            fill_start = i
            fill_end = i + 1
            
        for fill_i in range(fill_start, fill_end):
            if fill_i >= n_periods:
                break
                
            i_start = fill_i - lookback
            
            try:
                # Niveaux basés sur la densité des prix
                vals = data_log.iloc[i_start+1: fill_i+1]['close'].to_numpy()
                
                if (len(vals) >= 5 and 
                    not np.isnan(atr.iloc[fill_i]) and 
                    np.isfinite(atr.iloc[fill_i]) and 
                    atr.iloc[fill_i] > 0):
                    
                    levels, _, _, _, _, _ = find_levels(
                        vals, atr.iloc[fill_i], 
                        first_w, atr_mult, prom_thresh,
                        min_data_points=max(5, len(vals) // 3)
                    )
                    density_levels[fill_i] = levels
                    
            except Exception as e:
                pass  # Continue silencieusement
            
            try:
                # Lignes de tendance
                candles = data_log.iloc[i_start: fill_i+1]
                
                if len(candles) >= 3:
                    support_coefs, resist_coefs = fit_trendlines_high_low(
                        candles['high'].values, 
                        candles['low'].values, 
                        candles['close'].values
                    )
                    
                    if (support_coefs and support_coefs[0] is not None and 
                        np.isfinite(support_coefs[0]) and np.isfinite(support_coefs[1])):
                        support_slopes[fill_i] = support_coefs[0]
                        support_intercepts[fill_i] = support_coefs[1]
                        
                    if (resist_coefs and resist_coefs[0] is not None and 
                        np.isfinite(resist_coefs[0]) and np.isfinite(resist_coefs[1])):
                        resist_slopes[fill_i] = resist_coefs[0]
                        resist_intercepts[fill_i] = resist_coefs[1]
                        
            except Exception as e:
                pass  # Continue silencieusement
            
            # Calculer les points de pivot pour cette fenêtre
            try:
                window_data = data.iloc[i_start:fill_i+1]
                if len(window_data) >= 3:
                    pivot_window = max(1, min(3, len(window_data) // 5))
                    highs, lows = find_pivot_points(
                        window_data['high'].values, 
                        window_data['low'].values, 
                        window=pivot_window
                    )
                    
                    # Ajuster les indices pour correspondre aux données globales
                    for idx, price in highs:
                        pivot_highs_all.append((i_start + idx, price))
                    for idx, price in lows:
                        pivot_lows_all.append((i_start + idx, price))
                        
            except Exception as e:
                pass  # Continue silencieusement
    
    # Remplir les dernières valeurs par propagation si nécessaire
    for i in range(max(0, n_periods - step_size), n_periods):
        if density_levels[i] is None and i > 0:
            # Trouver la dernière valeur valide
            for j in range(i-1, -1, -1):
                if density_levels[j] is not None:
                    density_levels[i] = density_levels[j]
                    break
                    
        if np.isnan(support_slopes[i]) and i > 0:
            for j in range(i-1, -1, -1):
                if not np.isnan(support_slopes[j]):
                    support_slopes[i] = support_slopes[j]
                    support_intercepts[i] = support_intercepts[j]
                    break
                    
        if np.isnan(resist_slopes[i]) and i > 0:
            for j in range(i-1, -1, -1):
                if not np.isnan(resist_slopes[j]):
                    resist_slopes[i] = resist_slopes[j]
                    resist_intercepts[i] = resist_intercepts[j]
                    break
    
    return {
        'density_levels': density_levels,
        'support_slopes': support_slopes,
        'resist_slopes': resist_slopes,
        'support_intercepts': support_intercepts,
        'resist_intercepts': resist_intercepts,
        'pivot_highs': pivot_highs_all,
        'pivot_lows': pivot_lows_all,
        'atr_values': atr.values,
        'lookback_used': lookback
    }


def plot_professional_chart(data: pd.DataFrame, sr_results: dict, lookback: int, 
                          start_idx: int = None, end_idx: int = None, 
                          title: str = "Support/Resistance Analysis"):
    """Crée un graphique professionnel amélioré avec gestion d'erreurs"""
    
    if start_idx is None:
        start_idx = max(0, len(data) - min(300, len(data) // 2))
    if end_idx is None:
        end_idx = len(data)
    
    # Validation des indices
    start_idx = max(0, min(start_idx, len(data) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(data)))
    
    # Préparer les données pour l'affichage
    plot_data = data.iloc[start_idx:end_idx].copy()
    
    if len(plot_data) == 0:
        print("Pas de données à afficher")
        return None
    
    # Configuration du style
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Couleurs professionnelles
    bg_color = '#f8f9fa'
    grid_color = '#e9ecef'
    text_color = '#495057'
    support_color = '#28a745'
    resistance_color = '#dc3545'
    price_color = '#17a2b8'
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Prix de clôture principal
    ax.plot(plot_data.index, plot_data['close'], 
           color=price_color, linewidth=2, alpha=0.8, label=f'{title.split()[0]} Close')
    
    # Zones de prix (high/low) en gris transparent
    ax.fill_between(plot_data.index, plot_data['high'], plot_data['low'], 
                   alpha=0.1, color='gray', label='Price Range')
    
    # Affichage des niveaux de densité horizontaux
    if end_idx-1 < len(sr_results['density_levels']):
        current_levels = sr_results['density_levels'][end_idx-1]
        if current_levels and len(current_levels) > 0:
            current_price = plot_data['close'].iloc[-1]
            for i, level in enumerate(current_levels[:10]):  # Limiter à 10 niveaux
                color = support_color if level < current_price else resistance_color
                linestyle = '--' if level < current_price else '-.'
                
                ax.axhline(y=level, color=color, alpha=0.7, linestyle=linestyle, 
                          linewidth=2, label=f'S/R Level {level:.0f}' if i < 3 else "")
    
    # Lignes de tendance avec validation
    x_range = np.arange(len(plot_data))
    lines_plotted = 0
    max_lines = 3  # Réduire pour éviter l'encombrement
    
    # Parcourir les dernières lignes calculées
    step_back = max(10, lookback // 5)
    for i in range(end_idx-1, max(lookback, start_idx + 20), -step_back):
        if lines_plotted >= max_lines or i >= len(sr_results['support_slopes']):
            break
            
        if (i < len(sr_results['support_slopes']) and 
            i < len(sr_results['resist_slopes']) and
            not np.isnan(sr_results['support_slopes'][i]) and 
            not np.isnan(sr_results['resist_slopes'][i])):
            
            alpha = 0.8 - (lines_plotted * 0.2)
            
            try:
                # Ligne de support
                support_line = (sr_results['support_slopes'][i] * x_range + 
                              sr_results['support_intercepts'][i])
                support_line = np.exp(support_line)  # Convertir du log
                
                if np.all(np.isfinite(support_line)):
                    ax.plot(plot_data.index, support_line, color=support_color, 
                           alpha=alpha, linewidth=2.5, 
                           label='Support Trendline' if lines_plotted == 0 else "")
                
                # Ligne de résistance
                resist_line = (sr_results['resist_slopes'][i] * x_range + 
                             sr_results['resist_intercepts'][i])
                resist_line = np.exp(resist_line)  # Convertir du log
                
                if np.all(np.isfinite(resist_line)):
                    ax.plot(plot_data.index, resist_line, color=resistance_color, 
                           alpha=alpha, linewidth=2.5, 
                           label='Resistance Trendline' if lines_plotted == 0 else "")
                
                lines_plotted += 1
                
            except Exception as e:
                continue  # Passer à la ligne suivante en cas d'erreur
    
    # Points de pivot récents avec validation
    recent_highs = [(idx, price) for idx, price in sr_results['pivot_highs'] 
                   if start_idx <= idx < end_idx and np.isfinite(price)]
    recent_lows = [(idx, price) for idx, price in sr_results['pivot_lows'] 
                  if start_idx <= idx < end_idx and np.isfinite(price)]
    
    # Afficher seulement les pivots les plus significatifs
    if recent_highs:
        recent_highs = recent_highs[-15:]  # Réduire le nombre
        for idx, price in recent_highs:
            if idx < len(data):
                ax.scatter(data.index[idx], price, color=resistance_color, 
                          s=60, alpha=0.8, marker='v', zorder=5)
    
    if recent_lows:
        recent_lows = recent_lows[-15:]  # Réduire le nombre
        for idx, price in recent_lows:
            if idx < len(data):
                ax.scatter(data.index[idx], price, color=support_color, 
                          s=60, alpha=0.8, marker='^', zorder=5)
    
    # Configuration du graphique
    ax.set_title(f'{title} - Lookback: {lookback} periods', 
                fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_ylabel('Price', fontsize=12, color=text_color)
    ax.set_xlabel('Date', fontsize=12, color=text_color)
    
    # Grille professionnelle
    ax.grid(True, alpha=0.3, color=grid_color, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Formatage des dates sur l'axe x
    if len(plot_data) > 60:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(plot_data)//60)))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, len(plot_data)//20)))
        
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Légende professionnelle
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Ajout d'informations sur le graphique
    try:
        current_price = plot_data['close'].iloc[-1]
        if len(plot_data) > 1:
            price_change = plot_data['close'].iloc[-1] - plot_data['close'].iloc[-2]
            price_change_pct = (price_change / plot_data['close'].iloc[-2]) * 100
            info_text = f"Current: {current_price:.2f} ({price_change:+.2f}, {price_change_pct:+.2f}%)"
        else:
            info_text = f"Current: {current_price:.2f}"
            
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8), color=text_color)
    except:
        pass  # Ignorer les erreurs d'affichage
    
    plt.tight_layout()
    return fig


def plot_comparison_dashboard(data: pd.DataFrame, results_dict: dict, 
                             asset_name: str = "Asset"):
    """Crée un tableau de bord comparatif amélioré pour différents lookbacks"""
    
    lookbacks = list(results_dict.keys())
    n_plots = len(lookbacks)
    
    if n_plots == 0:
        print("Aucun résultat à afficher")
        return None
    
    # Organiser les subplots de manière adaptative
    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots == 2:
        rows, cols = 1, 2
    elif n_plots <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    plt.style.use('default')
    fig.patch.set_facecolor('#f8f9fa')
    
    for i, lookback in enumerate(lookbacks):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Données récentes pour l'affichage
        display_length = min(200, len(data) // 2)
        recent_data = data.tail(display_length)
        sr_data = results_dict[lookback]
        
        try:
            # Prix principal
            ax.plot(recent_data.index, recent_data['close'], 
                   color='#17a2b8', linewidth=2, label=f'{asset_name} Close')
            
            # Derniers niveaux de support/résistance
            if len(sr_data['density_levels']) > 0:
                last_levels = sr_data['density_levels'][-1]
                if last_levels and len(last_levels) > 0:
                    current_price = recent_data['close'].iloc[-1]
                    for level in last_levels[:5]:  # Limiter à 5 niveaux
                        if np.isfinite(level):
                            color = '#28a745' if level < current_price else '#dc3545'
                            ax.axhline(y=level, color=color, alpha=0.6, 
                                      linestyle='--', linewidth=1.5)
            
            # Configuration de chaque subplot
            ax.set_title(f'{asset_name} - Lookback: {lookback} periods', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Formatage des dates adaptatif
            if len(recent_data) > 60:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Erreur: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    # Masquer les subplots non utilisés
    for i in range(len(lookbacks), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{asset_name} - Support/Resistance Analysis Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def analyze_multiple_timeframes(data: pd.DataFrame, 
                               lookbacks: list = [20, 50, 100, 200], 
                               asset_name: str = "Asset"):
    """Analyse complète avec gestion d'erreurs améliorée"""
    
    # Validation et ajustement des lookbacks
    max_lookback = len(data) // 3
    valid_lookbacks = []
    
    for lb in lookbacks:
        if lb < 5:
            print(f"Warning: Lookback {lb} trop petit, ajusté à 5")
            lb = 5
        if lb > max_lookback:
            print(f"Warning: Lookback {lb} trop grand pour {len(data)} données, ajusté à {max_lookback}")
            lb = max_lookback
        if lb not in valid_lookbacks:
            valid_lookbacks.append(lb)
    
    lookbacks = sorted(valid_lookbacks)
    
    if not lookbacks:
        print("Aucun lookback valide trouvé")
        return {}
    
    results = {}
    
    print(f"=== Analyse de {asset_name} ===")
    print(f"Période: {data.index[0].strftime('%Y-%m-%d')} à {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Nombre de périodes: {len(data)}")
    print(f"Lookbacks ajustés: {lookbacks}")
    print()
    
    # Calcul pour chaque lookback
    for lookback in lookbacks:
        try:
            print(f"Calcul des niveaux S/R pour lookback = {lookback}...")
            sr_results = calculate_sr_levels(data, lookback)
            results[lookback] = sr_results
            
            # Graphique détaillé pour chaque lookback
            fig = plot_professional_chart(data, sr_results, lookback, 
                                         title=f"{asset_name} S/R Analysis")
            if fig:
                plt.show()
            else:
                print(f"Impossible de créer le graphique pour lookback {lookback}")
                
        except Exception as e:
            print(f"Erreur pour lookback {lookback}: {e}")
            continue
    
    # Tableau de bord comparatif si plusieurs résultats
    if len(results) > 1:
        try:
            print("\nCréation du tableau de bord comparatif...")
            comparison_fig = plot_comparison_dashboard(data, results, asset_name)
            if comparison_fig:
                plt.show()
        except Exception as e:
            print(f"Erreur lors de la création du tableau de bord: {e}")
    
    return results


if __name__ == '__main__':
    # Charger les données depuis Excel
    try:
        data = pd.read_excel('data_indices_OHLC_daily.xlsx', sheet_name='NDX', 
                           usecols=['TRADEDATE','Open','HIGH','LOW','LASTPRICE'])
        data.columns = ['date','open','high','low','close']
        data.set_index('date', inplace=True)
        data.index = pd.to_datetime(data.index)
        
        # Nettoyage des données amélioré
        print("=== NETTOYAGE DES DONNÉES ===")
        print(f"Valeurs nulles avant nettoyage: {data.isnull().sum().sum()}")
        print(f"Valeurs <= 0 avant nettoyage: {(data <= 0).sum().sum()}")
        
        # Supprimer les lignes avec des valeurs nulles ou <= 0
        data = data.dropna()
        data = data[(data > 0).all(axis=1)]
        
        # Vérifier la cohérence OHLC
        invalid_ohlc = ((data['high'] < data['low']) | 
                       (data['close'] > data['high']) | 
                       (data['close'] < data['low']) | 
                       (data['open'] > data['high']) | 
                       (data['open'] < data['low']))
        
        if invalid_ohlc.any():
            print(f"Warning: {invalid_ohlc.sum()} lignes avec OHLC incohérent supprimées")
            data = data[~invalid_ohlc]
        
        print(f"Nombre de bougies après nettoyage: {len(data)}")
        print(f"Valeurs nulles après nettoyage: {data.isnull().sum().sum()}")
        print(f"Valeurs <= 0 après nettoyage: {(data <= 0).sum().sum()}")
        print()
        
        # Analyse avec visualisations professionnelles
        # Utiliser des lookbacks plus petits pour tester les améliorations
        lookbacks = [10, 20, 50, 100]  # Inclut des lookbacks plus petits
        results = analyze_multiple_timeframes(data, lookbacks, "NASDAQ-100")
        
        print("=== ANALYSE TERMINÉE ===")
        print("Tous les graphiques ont été générés avec succès!")
        
    except FileNotFoundError:
        print("❌ Erreur: Le fichier 'data_indices_OHLC_daily.xlsx' n'a pas été trouvé.")
        print("Assurez-vous que le fichier est présent dans le répertoire courant.")
        
        # Exemple avec des données simulées
        print("\n=== GÉNÉRATION DE DONNÉES D'EXEMPLE ===")
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        
        # Simulation d'un mouvement de prix réaliste
        returns = np.random.normal(0.0008, 0.02, len(dates))
        prices = 1000 * np.exp(np.cumsum(returns))
        
        # Ajouter de la volatilité pour high/low
        volatility = np.random.normal(0, 0.01, len(dates))
        
        example_data = pd.DataFrame({
            'open': prices * (1 + volatility * 0.5),
            'high': prices * (1 + np.abs(volatility)),
            'low': prices * (1 - np.abs(volatility)),
            'close': prices
        }, index=dates)
        
        # Supprimer les weekends
        example_data = example_data[example_data.index.dayofweek < 5]
        
        print(f"Données d'exemple générées: {len(example_data)} points")
        print("Lancement de l'analyse sur les données d'exemple...")
        
        # Tester avec des lookbacks très petits pour valider les améliorations
        lookbacks = [5, 10, 20, 30]
        results = analyze_multiple_timeframes(example_data, lookbacks, "Example Asset")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        print("Vérifiez le format du fichier Excel et les noms des colonnes.")