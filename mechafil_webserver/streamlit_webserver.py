#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import streamlit.components.v1 as components
import st_debug as d

from datetime import date, timedelta, datetime
import time

import mechafil
from mechafil.data import (
    get_historical_network_stats, 
    get_sector_expiration_stats, 
    setup_spacescope,
    get_storage_baseline_value
)
from mechafil.power import (
    forecast_power_stats,
    build_full_power_stats_df,
    scalar_or_vector_to_vector,
)
from mechafil.vesting import compute_vesting_trajectory_df
from mechafil.minting import compute_minting_trajectory_df, compute_baseline_power_array
from mechafil.supply import forecast_circulating_supply_df

from mechafil.utils import validate_qap_method

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("debug.css")

def setup_data_access(bearer_token_or_cfg: str):
    setup_spacescope(bearer_token_or_cfg)

def validate_current_date(current_date: datetime.date):
    if current_date > (date.today() - timedelta(days=2)):
        raise ValueError("Current date must be at least 2 days in the past!")
    
init_baseline_global = 0.0
run_forecast_once_global = False
@st.cache_data
def cache_get_baseline_power_array(start_date, end_date):
    return compute_baseline_power_array(start_date, end_date)

@st.cache_data
def get_historical_data(start_date, current_date, end_date):
    global init_baseline_global
    sector_expiration_stats = get_sector_expiration_stats(start_date, current_date, end_date)
    fil_stats_df = get_historical_network_stats(start_date, current_date, end_date)
    init_baseline = get_storage_baseline_value(start_date)
    init_baseline_global = init_baseline
    return sector_expiration_stats, fil_stats_df

def run_simple_sim_cached(
    start_date: datetime.date,
    current_date: datetime.date,
    forecast_length: int,
    renewal_rate: Union[np.array, float],
    rb_onboard_power: Union[np.array, float],
    fil_plus_rate: Union[np.array, float],
    duration: int,
    bearer_token_or_cfg: str,
    qap_method: str = 'basic' # can be set to tunable or basic
                              # see: https://hackmd.io/O6HmAb--SgmxkjLWSpbN_A?view
) -> pd.DataFrame:
    global run_forecast_once_global
    validate_qap_method(qap_method)
    validate_current_date(current_date)
    
    end_date = current_date + timedelta(days=forecast_length)
    t1 = time.time()
    sector_expiration_stats, fil_stats_df = get_historical_data(start_date, current_date, end_date)
    baseline_power_array = cache_get_baseline_power_array(start_date, end_date)
    t2 = time.time()
    d.debug(f"get_historical_data took {t2-t1} seconds")

    # Get sector scheduled expirations
    # res = get_sector_expiration_stats(start_date, current_date, end_date)
    rb_known_scheduled_expire_vec = sector_expiration_stats[0]
    qa_known_scheduled_expire_vec = sector_expiration_stats[1]
    known_scheduled_pledge_release_full_vec = sector_expiration_stats[2]
    # Get daily stats
    # fil_stats_df = get_historical_network_stats(start_date, current_date, end_date)
    current_day_stats = fil_stats_df[fil_stats_df["date"] >= current_date].iloc[0]
    # Forecast power stats
    rb_power_zero = current_day_stats["total_raw_power_eib"] * 1024.0
    qa_power_zero = current_day_stats["total_qa_power_eib"] * 1024.0
    rb_power_df, qa_power_df = forecast_power_stats(
        rb_power_zero,
        qa_power_zero,
        rb_onboard_power,
        rb_known_scheduled_expire_vec,
        qa_known_scheduled_expire_vec,
        renewal_rate,
        fil_plus_rate,
        duration,
        forecast_length,
        qap_method=qap_method
    )
    rb_power_df["total_raw_power_eib"] = rb_power_df["total_power"] / 1024.0
    qa_power_df["total_qa_power_eib"] = qa_power_df["total_power"] / 1024.0
    power_df = build_full_power_stats_df(
        fil_stats_df,
        rb_power_df,
        qa_power_df,
        start_date,
        current_date,
        end_date,
    )
    # Forecast Vesting
    vest_df = compute_vesting_trajectory_df(start_date, end_date)
    # Forecast minting stats and baseline
    rb_total_power_eib = power_df["total_raw_power_eib"].values
    qa_total_power_eib = power_df["total_qa_power_eib"].values
    qa_day_onboarded_power_pib = power_df["day_onboarded_qa_power_pib"].values
    qa_day_renewed_power_pib = power_df["day_renewed_qa_power_pib"].values
    mint_df = compute_minting_trajectory_df(
        start_date,
        end_date,
        rb_total_power_eib,
        qa_total_power_eib,
        qa_day_onboarded_power_pib,
        qa_day_renewed_power_pib,
        baseline_power_array=baseline_power_array
    )
    # Forecast circulating supply
    start_day_stats = fil_stats_df.iloc[0]
    circ_supply_zero = start_day_stats["circulating_fil"]
    locked_fil_zero = start_day_stats["locked_fil"]
    daily_burnt_fil = fil_stats_df["burnt_fil"].diff().mean()
    burnt_fil_vec = fil_stats_df["burnt_fil"].values
    forecast_renewal_rate_vec = scalar_or_vector_to_vector(
        renewal_rate, forecast_length
    )
    past_renewal_rate_vec = fil_stats_df["rb_renewal_rate"].values[:-1]
    renewal_rate_vec = np.concatenate(
        [past_renewal_rate_vec, forecast_renewal_rate_vec]
    )
    cil_df = forecast_circulating_supply_df(
        start_date,
        current_date,
        end_date,
        circ_supply_zero,
        locked_fil_zero,
        daily_burnt_fil,
        duration,
        renewal_rate_vec,
        burnt_fil_vec,
        vest_df,
        mint_df,
        known_scheduled_pledge_release_full_vec,
    )
    run_forecast_once_global = True
    return cil_df

def my_melt(df_historical, df_forecast, col_name, var_name='horf', value_name='FIL'):
    df = df_historical[['date', col_name]]
    df.rename(columns={col_name: 'Historical'}, inplace=True)
    df = pd.concat([df, df_forecast[['date', col_name]]], ignore_index=True)
    df.rename(columns={col_name: 'Forecast'}, inplace=True)
    df = df.melt(id_vars=['date'], 
                 value_vars=['Historical', 'Forecast'], 
                 var_name=var_name, value_name=value_name)
    return df

def plot_panel(cil_df, simulation_start_date_obj):
    EIB=2**60
    # with st.empty():
        # fig = plt.figure(figsize=(12,12))
    # grid = plt.GridSpec(3, 2, wspace=0.3, hspace=0.3, figsize=(12,12))
    cil_df_plot = cil_df.iloc[1:]
    cil_df_plot.rename(columns={'network_baseline': 'Baseline', 'network_RBP': 'RBP', 'network_QAP': 'QAP'}, inplace=True)
    cil_df_historical = cil_df_plot[pd.to_datetime(cil_df_plot['date']) < pd.to_datetime(simulation_start_date_obj)]
    cil_df_forecast = cil_df_plot[pd.to_datetime(cil_df_plot['date']) >= pd.to_datetime(simulation_start_date_obj)]
    col1, col2 = st.columns(2)

    with col1:
        power_df = pd.melt(cil_df_plot, id_vars=["date"], 
                           value_vars=["Baseline", "RBP", "QAP"], var_name='Power', value_name='EIB')
        power_df['EIB'] = power_df['EIB'] / EIB
        power = (
            alt.Chart(power_df)
            .mark_line()
            .encode(x="date", y="EIB", color=alt.Color('Power', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Network Power")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(power.interactive(), use_container_width=True) 

        roi_df = my_melt(cil_df_historical, cil_df_forecast, '1y_sector_roi', value_name='ROI')
        roi = (
            alt.Chart(roi_df)
            .mark_line()
            .encode(x="date", y="ROI", color=alt.Color('horf', legend=alt.Legend(orient="top", title=None)))
            .properties(title="1Y Sector ROI")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(roi.interactive(), use_container_width=True)

    with col2:
        pledge_per_qap_df = my_melt(cil_df_historical, cil_df_forecast, 'day_pledge_per_QAP')
        day_pledge_per_QAP = (
            alt.Chart(pledge_per_qap_df)
            .mark_line()
            .encode(x="date", y="FIL", color=alt.Color('horf', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Pledge/32GiB QAP")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(day_pledge_per_QAP.interactive(), use_container_width=True)

        reward_per_sector_df = my_melt(cil_df_historical, cil_df_forecast, 'day_rewards_per_TiB')
        reward_per_sector = (
            alt.Chart(reward_per_sector_df)
            .mark_line()
            .encode(x="date", y="FIL", color=alt.Color('horf', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Reward/TiB")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(reward_per_sector.interactive(), use_container_width=True)


def forecast_economy():
    t1 = time.time()
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'

    rb_onboard_power_pib_day =  st.session_state['rbp_slider']
    renewal_rate_pct = st.session_state['rr_slider']
    fil_plus_rate_pct = st.session_state['fpr_slider']

    simulation_start_date = "today"
    forecast_length_days = 365*2  # gets us a 1Y ROI forecast
    sector_duration_days = 360

    start_date = date(2021, 3, 16)  # network start date
    if simulation_start_date == "today":
        current_date = date.today() - timedelta(days=3)
    else:
        try:
            current_date = date.fromisoformat(simulation_start_date)
        except:
            raise Exception("Simulation Start Date must be provided in ISO Format: YYYY-MM-DD")
    qap_method = 'basic'

    # scale from percentage to decimal
    renewal_rate = renewal_rate_pct / 100.
    fil_plus_rate = fil_plus_rate_pct / 100.

    setup_data_access(PUBLIC_AUTH_TOKEN)

    # # get historical data cached first
    # end_date = current_date + timedelta(days=forecast_length_days)
    # t2 = time.time()
    # get_historical_data(start_date, current_date, end_date)
    # cache_get_baseline_power_array(start_date, current_date, end_date)
    # t3 = time.time()
    # d.debug(f"Time to get historical data: {t3-t2}")

    cil_df = run_simple_sim_cached(
        start_date,
        current_date,
        forecast_length_days,
        renewal_rate,
        rb_onboard_power_pib_day,
        fil_plus_rate,
        sector_duration_days,
        PUBLIC_AUTH_TOKEN,
        qap_method=qap_method
    )

    # add generated quantities
    GIB = 2 ** 30
    SECTOR_SIZE = 32 * GIB
    PIB = 2 ** 50
    TIB = 2 ** 40
    cil_df['day_pledge_per_QAP'] = SECTOR_SIZE * (cil_df['day_locked_pledge']-cil_df['day_renewed_pledge'])/(cil_df['day_onboarded_power_QAP'])
    cil_df['day_rewards_per_sector'] = SECTOR_SIZE * cil_df.day_network_reward / cil_df.network_QAP
    
    cil_df['day_rewards_per_TiB'] = (cil_df['day_rewards_per_sector'] / SECTOR_SIZE) * TIB

    cil_df['1y_return_per_sector'] = cil_df['day_rewards_per_sector'].rolling(365).sum().shift(-365+1).values.flatten()
    cil_df['1y_sector_roi'] = cil_df['1y_return_per_sector'] / cil_df['day_pledge_per_QAP']

    cil_df['date'] = pd.to_datetime(cil_df['date'])

    # st.write(cil_df)
    t4 = time.time()
    d.debug(f"Time to forecast: {t4-t1}")
    plot_panel(cil_df, current_date)

def main():
    st.title('Filecoin Economy Forecaster')

    st.slider("Raw Byte Onboarding (PiB/day)", min_value=3., max_value=20., value=6., step=.1, format='%0.02f', key="rbp_slider",
              on_change=None, kwargs=None, disabled=False, label_visibility="visible")
    st.slider("Renewal Rate (Percentage)", min_value=10, max_value=99, value=60, step=1, format='%d', key="rr_slider",
              on_change=None, kwargs=None, disabled=False, label_visibility="visible")
    st.slider("FIL+ Rate (Percentage)", min_value=10, max_value=99, value=70, step=1, format='%d', key="fpr_slider",
              on_change=None, kwargs=None, disabled=False, label_visibility="visible")
    st.button("Forecast", on_click=forecast_economy)

    # forecast_economy()

    if "debug_string" in st.session_state:
        st.markdown(
            f'<div class="debug">{ st.session_state["debug_string"]}</div>',
            unsafe_allow_html=True,
        )
    components.html(
        d.js_code(),
        height=0,
        width=0,
    )

if __name__ == '__main__':
    main()