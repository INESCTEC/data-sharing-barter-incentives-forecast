# flake8: noqa
import os
import streamlit as st
from util.util_plot import plot_boxplot, plot_session_stack_bar, plot_avg_barline

__ROOT_DIR__ = os.path.dirname(os.path.dirname(__file__))
__DEFAULT_REPORTS_DIR__ = os.path.join(__ROOT_DIR__, "files", "reports")
__DEFAULT_DATASET_DIR__ = os.path.join(__ROOT_DIR__, "files", "datasets")

print(os.path.exists(__DEFAULT_REPORTS_DIR__))


available_reports = []
for x in os.listdir(__DEFAULT_REPORTS_DIR__):
    # list subdirs:
    for y in os.listdir(os.path.join(__DEFAULT_REPORTS_DIR__, x)):
        # list files:
        available_reports.append(os.path.join(x, y))

available_datasets = []
for x in os.listdir(__DEFAULT_DATASET_DIR__):
    available_datasets.append(x)

# app markdown setting
st.set_page_config(layout="wide")
style = "<style>h1 {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)
style = "<style>h2 {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)
style = "<style>h3 {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)

st.header('Predico-Data Analytics')
st.subheader('Standard')
path_report = st.selectbox(
    'Enter the folder path for STANDARD REPORT',
    available_reports,
    key="path_report")
report_dataset_ref = os.path.split(path_report)[0]
path_report = os.path.join(__DEFAULT_REPORTS_DIR__, path_report)
path_dataset = os.path.join(__DEFAULT_DATASET_DIR__, report_dataset_ref)
st.write('Report:', path_report)
st.write('Dataset:', path_dataset)

st.subheader('Feature Selection')
path_report_fs = st.selectbox(
    'Enter the folder path for STANDARD REPORT',
    available_reports,
    key="path_report_fs")
report_dataset_fs_ref = os.path.split(path_report_fs)[0]
path_report_fs = os.path.join(__DEFAULT_REPORTS_DIR__, path_report_fs)
path_dataset_fs = os.path.join(__DEFAULT_DATASET_DIR__, report_dataset_fs_ref)
st.write('Report:', path_report_fs)
st.write('Dataset:', path_dataset_fs)

if (path_report) and (path_dataset) and (path_report_fs) and (path_dataset_fs):

    st.divider()
    with st.expander('MAE-based Forecasting Accuracy'):

        st.header('MAE-based Forecasting Accuracy')

        from util.util import melt_results
        from util.util_forecast import forecast_performance

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Standard')

            path_prediction = os.path.join(path_report, 'forecasts.csv')
            path_target = os.path.join(path_dataset, 'measurements.csv')

            df_results = forecast_performance(path_prediction, path_target)
            df_melt = melt_results(df_results)

            # plots
            fig = plot_session_stack_bar(df_melt, "session_id", "performance_error", "resource_id", "MAE per session")
            st.plotly_chart(fig, theme="streamlit")

            fig = plot_boxplot(df_melt, "resource_id", "performance_error", "resource_id", "Boxplot MAE")
            st.plotly_chart(fig, theme="streamlit")

        with col2:
            st.subheader('Feature Selection')

            path_prediction = os.path.join(path_report_fs, 'forecasts.csv')
            path_target = os.path.join(path_dataset_fs, 'measurements.csv')

            df_results = forecast_performance(path_prediction, path_target)
            df_melt = melt_results(df_results)

            # plots
            fig = plot_session_stack_bar(df_melt, "session_id", "performance_error", "resource_id", "MAE per session")
            st.plotly_chart(fig, theme="streamlit")
            fig = plot_boxplot(df_melt, "resource_id", "performance_error", "resource_id", "Boxplot MAE")
            st.plotly_chart(fig, theme="streamlit")

    try:
        # Revenue-Coefficients matching
        with st.expander('Sellers Revenue-Coefficients Matching'):
            from util.util_revenue import revenue_avg, revenue_session

            st.subheader('Sellers Revenue-Coefficients Matching')
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Standard')

                path_coefs = os.path.join(path_dataset, 'new_coefs.csv')
                path_sellers = os.path.join(path_report, 'sellers.csv')

                df_revenue_avg = revenue_avg(path_coefs, path_sellers)
                df_revenue_session = revenue_session(path_coefs, path_sellers)

                # plots
                fig = plot_session_stack_bar(df_revenue_session, "session_id", "difference", "resource_id", "Coefs-Revenue Difference per session")
                st.plotly_chart(fig, theme="streamlit")

                fig = plot_boxplot(df_revenue_session,"resource_id", "difference", "resource_id", "Coefs-Revenue Difference")
                st.plotly_chart(fig, theme="streamlit")

                fig = plot_avg_barline(df_revenue_avg, "resource_id", "norm_to_receive", "norm_coefs", "difference", "Coefs-Revenue Difference")
                st.plotly_chart(fig, theme="streamlit")

            with col2:
                st.subheader('Feature Selection')

                path_coefs = os.path.join(path_dataset_fs, 'new_coefs.csv')
                path_sellers = os.path.join(path_report_fs, 'sellers.csv')

                df_revenue_avg = revenue_avg(path_coefs, path_sellers)
                df_revenue_session = revenue_session(path_coefs, path_sellers)

                # plots
                fig = plot_session_stack_bar(df_revenue_session, "session_id", "difference", "resource_id", "Coefs-Revenue Difference per session")
                st.plotly_chart(fig, theme="streamlit")

                fig = plot_boxplot(df_revenue_session,"resource_id", "difference", "resource_id", "Coefs-Revenue Difference")
                st.plotly_chart(fig, theme="streamlit")

                fig = plot_avg_barline(df_revenue_avg, "resource_id", "norm_to_receive", "norm_coefs", "difference", "Coefs-Revenue Difference")
                st.plotly_chart(fig, theme="streamlit")
    except FileNotFoundError:
        pass

    # Payment-Gain matching
    with st.expander('Buyers Payment-Gain Matching'):
        from util.util_payment import process_buyers_avg, process_payment_session

        st.subheader('Buyers Payment-Gain Matching')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Standard')

            path_buyers = os.path.join(path_report, 'buyers.csv')

            df_payment_session = process_payment_session(path_buyers)
            df_payment_avg = process_buyers_avg(path_buyers)

            # plots
            fig = plot_session_stack_bar(df_payment_session, "session_id", "difference", "resource_id", "Gain-Payment Difference per session")
            st.plotly_chart(fig, theme="streamlit")

            fig = plot_boxplot(df_payment_session,"resource_id", "difference", "resource_id", "Gain-Payment Difference")
            st.plotly_chart(fig, theme="streamlit")

            fig = plot_avg_barline(df_payment_avg, "resource_id", "norm_to_pay", "norm_gain", "difference", "Gain-Payment Difference per resource")
            st.plotly_chart(fig, theme="streamlit")

        with col2:
            st.subheader('Feature Selection')

            path_buyers = os.path.join(path_report_fs, 'buyers.csv')

            df_payment_session = process_payment_session(path_buyers)
            df_payment_avg = process_buyers_avg(path_buyers)

            # plots
            fig = plot_session_stack_bar(df_payment_session, "session_id", "difference", "resource_id", "Gain-Payment Difference per session")
            st.plotly_chart(fig, theme="streamlit")

            fig = plot_boxplot(df_payment_session,"resource_id", "difference", "resource_id", "Gain-Payment Difference")
            st.plotly_chart(fig, theme="streamlit")

            fig = plot_avg_barline(df_payment_avg, "resource_id", "norm_to_pay", "norm_gain", "difference", "Gain-Payment Difference per resource")
            st.plotly_chart(fig, theme="streamlit")

    with st.expander('Market Performance'):
        st.header('Market Performance')

        from util.util_performance import read_market_elapsed_time
        from util.util_plot import plot_performance

        path_std = os.path.join(path_report, 'market.csv')
        path_fs = os.path.join(path_report_fs, 'market.csv')
        df_performance = read_market_elapsed_time(path_std, path_fs)

        fig = plot_performance(df_performance, 'session_id', 'elaps_time_std', 'elaps_time_fs', 'Market Performance per Session')
        st.plotly_chart(fig, theme="streamlit")

