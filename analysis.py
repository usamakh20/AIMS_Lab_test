import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

covid_df = pd.read_csv("COVID-19.csv", keep_default_na=False)

# Pre processing
covid_df['day'] = pd.to_datetime(covid_df['day'])
covid_df = covid_df[covid_df['Country'] != '']

# Range of days in data
days = np.sort(covid_df['day'].unique())

# List of all countries in data
countries = np.sort(covid_df['Country Name'].unique())

# -----------------  Query 1 ------------------------

selected_day = days[int(len(days) / 2)]  # 27th Feb

day_stats = covid_df.loc[covid_df.day == selected_day]
day_stats_confirmed = day_stats.sort_values(by='Confirmed', ascending=False).loc[:,
                      ['day', 'Country Name', 'Confirmed']].head(20).query('Confirmed > 0')
day_stats_deaths = day_stats.sort_values(by='Deaths', ascending=False).loc[:, ['day', 'Country Name', 'Deaths']].head(
    20).query('Deaths > 0')

# Plot Top 20 countries with the most confirmed cases on 27th Feb
fig, ax = plt.subplots(figsize=(30, 15))
plt.title('Top 20 Countries with confirmed cases on 27th Feb', fontsize=40)
ax.set_xlabel('Country', fontsize=20)
ax.set_ylabel('Confirmed Cases', fontsize=20)
ax.bar(day_stats_confirmed['Country Name'].to_numpy(), day_stats_confirmed['Confirmed'].to_numpy(), 0.5)
plt.show()
fig.savefig('Query 1: Top 20 Countries with confirmed cases on 27th Feb.svg')

# Plot Top 20 countries with the most deaths on 27th Feb
fig, ax = plt.subplots(figsize=(10, 5))
plt.title('Top 20 Countries with most deaths on 27th Feb')
ax.set_xlabel('Countries')
ax.set_ylabel('Deaths')
ax.bar(day_stats_deaths['Country Name'].to_numpy(), day_stats_deaths['Deaths'].to_numpy(), 0.5, color='red')
plt.show()
fig.savefig('Query 1: Top 20 Countries with most deaths on 27th Feb.svg')

# Recovered cases cannot be calculated because column not present

# -----------------  Query 2 ------------------------

selected_date_1 = pd.to_datetime(days[int(len(days) / 4)])  # 2nd Feb
selected_date_2 = pd.to_datetime(days[3 * int(len(days) / 4)])  # 23rd March

cases_counter = Counter()
deaths_counter = Counter()
for country in countries:
    country_cases_deaths = covid_df[(covid_df['Country Name'] == country) & (covid_df['day'] > selected_date_1) & (
            covid_df['day'] < selected_date_2)].sort_values(by='day')[['Cumulative Confirmed', 'Cumulative Deaths']]

    if not country_cases_deaths.empty:
        country_cases = country_cases_deaths['Cumulative Confirmed']
        country_deaths = country_cases_deaths['Cumulative Deaths']

        if not country_cases.empty:
            cumulative = country_cases.iloc[[0, -1]].to_numpy()
            cases_counter[country] = cumulative[1] - cumulative[0]

        if not country_deaths.empty:
            cumulative = country_deaths.iloc[[0, -1]].to_numpy()
            deaths_counter[country] = cumulative[1] - cumulative[0]

most_cases = cases_counter.most_common(1)[0]
most_deaths = deaths_counter.most_common(1)[0]

print("Query 2: ")
print("Country with most cases between %s and %s is %s with %s cases" % (
selected_date_1.strftime('%d-%b'), selected_date_2.strftime('%d-%b'), most_cases[0], most_cases[1]))
print("Country with most deaths between %s and %s is %s with %s deaths" % (
selected_date_1.strftime('%d-%b'), selected_date_2.strftime('%d-%b'), most_deaths[0], most_deaths[1]))
print('----------------------------')

# -----------------  Query 3 ------------------------


selected_country = countries[0]  # Afghanistan
cases_span = covid_df[covid_df['Country Name'] == selected_country].loc[:, ['day', 'Confirmed']].sort_values(
    by='day').to_numpy()

prev_cases_highest = 0
spread = ['', 0]
non_spread = 0
spread_periods = Counter()
for day, new_cases in cases_span:
    if new_cases > prev_cases_highest:
        if spread[1] > 0:
            spread[1] += non_spread + 1
        else:
            spread[0] = day
            spread[1] += 1
        non_spread = 0
        prev_cases_highest = new_cases
    else:
        non_spread += 1

    if non_spread > 3:
        non_spread = 0
        if spread[1] > 0:
            spread_periods[spread[0]] = spread[1]
        spread[1] = 0
        spread[0] = ''
        prev_cases_highest = 0

if spread[1] > 0:
    spread_periods[spread[0]] = spread[1]

longest_spread = spread_periods.most_common(1)[0]
print('Query 3: Longest spread in %s was of %s days starting from %s' % (
selected_country, longest_spread[1], pd.to_datetime(longest_spread[0]).strftime('%d-%b')))
print('----------------------------', end='\n\n\n\n')

daily_cases = cases_span.transpose()
y = daily_cases[1].astype('int')
x = np.arange(len(daily_cases[0]))
plt.figure(figsize=(10, 5))
plt.title(selected_country + ' Daily new cases', fontsize=15)
plt.ylabel('New Cases', fontsize=10)
plt.xlabel('Days starting from ' + daily_cases[0][0].strftime('%d-%b'), fontsize=10)
plt.plot(x, y)
plt.show()

# -----------------  Query 4 ------------------------

country_case_count = Counter()
for country in countries:
    count_cases = covid_df[covid_df['Country Name'] == country].loc[:, ['day', 'Cumulative Confirmed']].sort_values(
        by='day').to_numpy()
    country_case_count[country] = count_cases[-1][1]

top5_countries = [country for country, cases in country_case_count.most_common(5)]

for country in top5_countries:
    country_cases = covid_df[covid_df['Country Name'] == country][['day', 'Confirmed']].sort_values(
        by='day').to_numpy().transpose()
    model = make_pipeline(PolynomialFeatures(4), Ridge())
    x = np.array(range(len(country_cases[1]))).reshape(-1, 1)
    model.fit(x, country_cases[1])
    x_pred_30 = np.array(range(len(country_cases[1]) + 30)).reshape(-1, 1)
    y_pred_30 = model.predict(x_pred_30)
    y_pred_30[y_pred_30 < 0] = 0
    fig = plt.figure()
    plt.title('30 days Prediction for cases in ' + country)
    plt.plot(x, country_cases[1], label='orignal')
    plt.plot(x_pred_30, y_pred_30, label='predicted')
    plt.ylabel('New Cases', fontsize=10)
    plt.xlabel('Days starting from ' + country_cases[0][0].strftime('%d-%b'), fontsize=10)
    plt.legend()
    fig.savefig('Query 4: 30 days Prediction for cases in ' + country + '.svg')
